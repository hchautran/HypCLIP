
import wandb
from accelerate import Accelerator
from model.hypCLIP import HypCLIP
from model.hypBLIP import HypBLIP 
from geoopt.optim import RiemannianAdam, RiemannianSGD
from utils.retrivial_utils import evaluate_recall 
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from config import EUCLID, LORENTZ, POINCARE
from model.hypBLIP  import HypBLIP
from model.hypCLIP import HypCLIP
from typing import Union
from copy import deepcopy



class MyTrainerWithMomentum(torch.nn.Module):
    def __init__(self, config, model:Union[HypBLIP, HypCLIP] ,dataset ,train_loader, val_loader, test_loader, processor):
        super().__init__()
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
        self.embed_dim = config.ft_out
        self.alpha = config.alpha 
        self.negative_all_rank = False 
        self.processor = processor 
        self.device = torch.device(f'cuda:{config.cuda}' if (torch.cuda.is_available() and config.cuda >=0) else 'cpu')
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision, 
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        self.enable_log = self.config.enable_log
        self.current_epoch = 0

        model.visual_encoder_m = deepcopy(model.vision_model)
        model.text_encoder_m = deepcopy(model.text_model)

        model.model_pairs = [
            [model.vision_model, model.visual_encoder_m],
            [model.text_model, model.text_encoder_m],
        ]

        model.register_buffer("image_queue", torch.randn(self.embed_dim, model.queue_size))
        model.register_buffer("text_queue", torch.randn(self.embed_dim, model.queue_size))
        model.register_buffer("idx_queue", torch.full((1, model.queue_size), -100))
        model.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.model = self.accelerator.prepare(model)
        self.model.copy_params()

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


    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

        
    def contrastive_loss(self, sims_i2t, sims_t2i, pos_mask):
        ones = torch.ones_like(sims_i2t).to(self.device)
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign =  ones.masked_fill(torch.eq(ones, pos_mask), -1.0)

        if self.config.manifold in [LORENTZ, POINCARE]:
            neg_margin = self.config.lorentz_neg_margin * neg_mask 
            pos_margin = self.config.lorentz_pos_margin * pos_mask 
            sims_i2t = sims_i2t + neg_margin 
            sims_t2i = sims_t2i + neg_margin 
            sims_i2t = (sims_i2t + pos_margin) * sign
            sims_t2i = (sims_t2i + pos_margin) * sign 
        else:
            neg_margin = self.config.euclid_neg_margin * neg_mask 
            pos_margin = self.config.euclid_pos_margin * pos_mask 
            sims_i2t = sims_i2t - neg_margin 
            sims_t2i = sims_t2i - neg_margin 
            sims_i2t = (sims_i2t - pos_margin) * sign 
            sims_t2i = (sims_t2i - pos_margin) * sign 

        sims = torch.cat([torch.clamp(sims_i2t, min=0.0), torch.clamp(sims_t2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss

    def itm_loss(self, imgs, cap, sim_i2t, sim_t2i, idx):
        # Find negative
        with torch.no_grad():
            bs = imgs.size(0)
            weights_i2t = F.softmax(sim_i2t+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i+1e-4,dim=1)
            mask = torch.eq(idx, idx.T)
            mask = mask.to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 
        # select a negative image for each text
        img_enc_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            img_enc_neg.append(imgs[neg_idx])
        img_enc_neg = torch.stack(img_enc_neg,dim=0) 

        # select a negative text for each image
        cap_enc_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            cap_enc_neg.append(cap[neg_idx])
        cap_enc_neg = torch.stack(cap_enc_neg,dim=0)   

        cap_enc_all = torch.cat([cap, cap, cap_enc_neg],dim=0)     
        img_enc_all = torch.cat([imgs, img_enc_neg, imgs],dim=0)
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.float),torch.zeros(2*bs,dtype=torch.float)],
                               dim=0).view(-1,1).to(imgs.device)

        disc = self.model.discriminator(img_enc_all, cap_enc_all)
        loss_itm = F.binary_cross_entropy(disc, itm_labels)
        return loss_itm

    def forward_batch(self, data, epoch, current_step):
        idx = data['img_id']
        alpha = self.alpha * self._rampup_factor(
            epoch=epoch,
            iters=current_step,
            num_iters_per_epoch=len(self.train_loader)
        )
        with torch.no_grad():
            self.model.temp.clamp_(0.001, 0.5)

        image_feat = self.model.get_vision_features(data['pixel_values'])
        text_feat = self.model.get_text_features(data['input_ids'], data['attention_mask'])
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.model.idx_queue.clone().detach()], dim=1)
        pos_mask = torch.eq(idx, idx_all).float().to(self.device)
        bsize = pos_mask.shape[0] 
        eye_mask = torch.eye(bsize).to(self.device) * 1e9
        sim_targets = (pos_mask /  pos_mask.sum(1, keepdim=True)).to(self.device)
        cur_sims_i2i = self.model.dist_func(image_feat, image_feat) - eye_mask
        cur_sims_i2t = self.model.dist_func(image_feat, text_feat) 
        with torch.no_grad():
            self.model._momentum_update()
            text_feat_m = self.model.text_encoder_m(data['input_ids'], data['attention_mask'])[1]
            image_feat_m = self.model.visual_encoder_m(data['pixel_values'])[1]
            text_feat_m_all = torch.cat([text_feat_m.t(), self.model.text_queue.clone().detach()], dim=1)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.model.image_queue.clone().detach()], dim=1)
            sim_i2t_m = self.model.dist_func(image_feat_m,  text_feat_m_all.T)
            sim_t2i_m = self.model.dist_func(text_feat_m,  image_feat_m_all.T)
            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sims_i2t = self.model.dist_func(image_feat, text_feat_m_all.T)
        sims_t2i = self.model.dist_func(text_feat, image_feat_m_all.T)

        loss_i2t = -torch.sum(
            F.log_softmax(sims_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sims_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()
        self.model._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        
        loss_itc = (loss_i2t + loss_t2i) / 2
        loss_contrastive = self.model.contrastive_loss(cur_sims_i2t, cur_sims_i2i)
        loss_itm = self.itm_loss(image_feat, text_feat, cur_sims_i2t, cur_sims_i2t.T, idx)
        loss = loss_itc + loss_contrastive + loss_itm 
        bs = sims_i2t.shape[0] 
        in_batch_acc = (cur_sims_i2t.argmax(-1) == torch.arange(bs).to(self.device)).float().mean().item()

                
        stats = {
            "logits/contrastive_loss": loss_contrastive.item(), 
            "logits/itm_loss": loss_itm.item(), 
            "logits/itc_loss": loss_itc.item(), 
            "logits/current_loss": loss.item(), 
            "logits/min": sims_i2t.min().item(),
            "logits/mean": sims_i2t.mean().item(),
            "logits/max": sims_i2t.max().item(),
            "logits/acc": in_batch_acc 
        }

        return loss,  stats



    def train(self):
        # loop over the dataset multiple times
        current_step = 0
        best_r_all = 0.0
        waiting = 0
        for epoch in range(self.epochs):
            with self.accelerator.accumulate(self.model):
                self.model.train()
                self.current_epoch = epoch

                running_loss = 0.0
                current_step = 0
                for data in tqdm(self.train_loader):
                    self.accelerator.free_memory()
                    self.optimizer.zero_grad()
                    current_step+=1
                    current_step += 1
                    loss, stats = self.forward_batch(data, epoch, current_step)
                    self.accelerator.backward(loss)
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    running_loss += loss.item()

                    if (current_step+1) % self.log_freq == 0:
                        self.log(stats)
                        self.log({
                            'curvature': self.model.curv.item(),
                            'temp': self.model.temp.item(),
                        })
                        print(stats)
                        print('Loss: {}'.format(loss.item()))
                metrics = self.evaluate()
                print(metrics)
                self.log(metrics)
                if best_r_all < metrics['r_all']:
                    waiting = 0
                    best_r_all = metrics['r_all']
                    self.log({'best r_all': metrics['r_all']})
                    print('best r all', best_r_all)
                elif epoch > self.config.min_epochs:
                    waiting += 1
                if waiting > self.patience:
                    break
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
        

    
