
from datasets import load_dataset
from transformers import CLIPProcessor, BlipProcessor 
import wandb
from accelerate import Accelerator
from model.hypCLIP import HypCLIP
from model.hypBLIP import HypBLIP 
from utils.data_utils import get_dataloader, preprocess_img
from geoopt.optim import RiemannianAdam, RiemannianSGD
from utils.retrivial_utils import evaluate_recall 
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from utils.data_utils import get_flickr
from config import EUCLID, LORENTZ, POINCARE


class MyTrainer():
    def __init__(self, config, model ,dataset ,train_loader, val_loader, test_loader, processor):
        self.config = config 
        self.model_ckt = config.model_ckt
        self.grad_clip = config.grad_clip
        self.min_epochs = config.min_epochs
        self.eval_freq = config.eval_freq
        self.log_freq = config.log_freq
        self.patience = config.patience
        self.save_dir = config.save_dir
        self.epochs = config.epochs
        self.cache_dir = config.cache_dir
        self.processor = processor 
        self.device = torch.device(f'cuda:{config.cuda}' if (torch.cuda.is_available() and config.cuda >=0) else 'cpu')
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision, 
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        self.enable_log = self.config.enable_log
        self.current_epoch = 0
        self.model = self.accelerator.prepare(model)
        self.dataset = dataset

        if config.manifold == EUCLID:
            if config.optimizer == 'adam':
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=config.lr, 
                    weight_decay=config.weight_decay,
                )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), 
                    lr=config.lr,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay,
                    
                )
        else:
            if config.optimizer == 'adam':
                self.optimizer = RiemannianAdam(
                    self.model.parameters(), 
                    lr=config.lr, 
                    stabilize=10, 
                    weight_decay=config.weight_decay
                )
            else:
                self.optimizer == RiemannianSGD(
                    self.model.parameters(),
                    momentum=config.momentum,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    stabilize=10
                )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.lr_reduce_freq,
            gamma=config.gamma

        )
        self.optimizer, self.train_loader, self.val_loader, self.test_loader, self.scheduler = self.accelerator.prepare(
            self.optimizer, train_loader, val_loader, test_loader ,self.scheduler
        )
        self.name=f'{config.model_ckt}_{config.manifold}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}_{config.batch_size}_{config.ft_out}'
        print('RUNNING:',self.name)


        if self.enable_log:
            wandb.init(
                name=self.name,
                config=vars(self.config),
                reinit=True
            )
        print('trainable parameters:', self.model.num_parameters())
        self.log({'trainable parameters': self.model.num_parameters()})
            
    def log(self, stat):
        if self.enable_log:
            wandb.log(stat)


    def train(self):
        # loop over the dataset multiple times
        current_step = 0
        best_r_all = 0.0
        waiting = 0
        
        for epoch in range(self.epochs):
            with self.accelerator.accumulate(self.model):
                self.current_epoch = epoch
                self.model.train()

                running_loss = 0.0
                for data in tqdm(self.train_loader):
                    self.accelerator.free_memory()
                    self.optimizer.zero_grad()
                    current_step += 1
                    # assert len(img_ids) == len(set(img_ids))
            
                    loss, stats, loss_itc, loss_itm = self.model(
                        input_ids=data['input_ids'],
                        attention_mask=data['attention_mask'],
                        pixel_values=data['pixel_values'],
                    )

                    self.accelerator.backward(loss)
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()

                    running_loss += loss.item()

                    if (current_step+1) % self.log_freq == 0:
                        self.log(stats)
                        self.log({
                            'current loss itm': loss_itm.item(),
                            'current loss itc': loss_itc.item(),
                            'current loss': loss.item(),
                            'curvature': self.model.curv.item(),
                            'temp': self.model.temp.item(),
                            
                        })
                        print(stats)
                        print('Loss: {}'.format(loss.item()))
                    # print('infer time', time.time() - start)
                metrics = self.evaluate()
                print(metrics)
                self.log(metrics)
                if best_r_all < metrics['r_all']:
                    waiting = 0
                    best_r_all = metrics['r_all']
                    print('best r all', best_r_all)
                else:
                    waiting += 1
                if waiting < self.patience:
                    self.train_loader = self.accelerator.prepare(
                        get_dataloader(self.dataset['train'], self.config.batch_size, processor=self.processor, mode='train')
                    )
                else: break
            
        print('Finished Training')

    def evaluate(self, mode='val'):
        print("Evaluating current epoch", self.current_epoch)
        self.model.eval()
        loader = self.val_loader if mode == 'val' else  self.test_loader
        all_text_embeds = [] 
        all_vision_embeds = [] 
        with torch.no_grad():
            for data in tqdm(loader):
                text_embeds = self.model.get_text_features(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
                vision_embeds = self.model.get_vision_features(pixel_values=data['pixel_values'][0].unsqueeze(0))
                all_text_embeds.append(text_embeds)
                all_vision_embeds.append(vision_embeds)

            all_text_embeds = torch.concat(all_text_embeds, 0) 
            all_vision_embeds = torch.concat(all_vision_embeds, 0) 
            sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds).cpu().detach().numpy()
            metrics = evaluate_recall(sims_t2i=sims_t2i)
            metrics['epoch'] = self.current_epoch
        return metrics


    def save(self):
        torch.save(self.model ,f'{self.save_dir}/{self.name}.pth')


    def load(self):
        self.model = torch.load(f'{self.save_dir}/{self.name}.pth')
        

    
