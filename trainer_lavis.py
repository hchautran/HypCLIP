import wandb
from accelerate import Accelerator
from utils.data_utils import get_dataloader
from hyptorch.geoopt.optim import RiemannianAdam, RiemannianSGD
from utils.retrivial_utils import evaluate_recall
from model.baseQueueModel import BaseModelWithQueue
from tqdm.auto import tqdm
import torch
from config import EUCLID, POINCARE, LORENTZ
import time



class MyTrainer:
    def __init__(
        self, config, model, dataset, train_loader, val_loader, test_loader, processor
    ):
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
        self.momentum = config.momentum
        self.processor = processor
        self.device = torch.device(
            f"cuda:{config.cuda}"
            if (torch.cuda.is_available() and config.cuda >= 0)
            else "cpu"
        )
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.enable_log = self.config.enable_log
        self.current_epoch = 0
        self.model = self.accelerator.prepare(model)
        self.dataset = dataset

        if config.manifold == EUCLID:
            if config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=config.lr,
                )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=config.lr,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay,
            )
        else:
            if config.optimizer == "adam":
                self.optimizer = RiemannianAdam(
                    self.model.parameters(),
                    lr=config.lr,
                    stabilize=10,
                    weight_decay=config.weight_decay,
                )
            else:
                self.optimizer = RiemannianSGD(
                    self.model.parameters(),
                    momentum=config.momentum,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    stabilize=10,
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.1, patience=2
        )
        
        (
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, train_loader, val_loader, test_loader, self.scheduler
        )
        self.name = f'{config.model_ckt.split("/")[-1]}_{config.manifold}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}_{config.batch_size}_{config.ft_out}'
        print("RUNNING:", self.name)

        if self.enable_log:
            wandb.init(name=self.name, config=vars(self.config), reinit=True, project="Graph")
        print("trainable parameters:", self.model.num_parameters())
        self.log({"trainable parameters": self.model.num_parameters()})

    def log(self, stat):
        if self.enable_log:
            wandb.log(stat)

    def train(self):
        # loop over the dataset multiple times
        best_r_all = 0.0
        waiting = 0
        for epoch in range(self.epochs):
            current_step = 0
            with self.accelerator.accumulate(self.model):
                self.model.train()
                self.current_epoch = epoch
                running_loss = 0.0
                print('train loader length:', len(self.train_loader))
                for data in tqdm(self.train_loader):
                    start = time.time()
                    self.accelerator.free_memory()
                    self.optimizer.zero_grad()
                    current_step += 1
                    # assert len(img_ids) == len(set(img_ids))

                    loss, stats = self.model(
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                        pixel_values=data["pixel_values"],
                        image_id=data['img_id'],
                        epoch=epoch,
                        iters=current_step,
                        num_iters_per_epoch=len(self.train_loader),
                    )

                    self.accelerator.backward(loss)
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()

                    running_loss += loss.item()

                    if (current_step + 1) % self.log_freq == 0:
                        self.log(stats)
                        print(stats)
                        print("Loss: {}".format(loss.item()))
                    if self.eval_freq != -1 and (current_step + 1) % self.eval_freq == 0:
                        metrics = self.evaluate(mode='val')
                        self.model.train()
                        print(metrics)
                        self.log(metrics)
                        self.scheduler.step(metrics["val/r_all"])
                    print('infer time', time.time() - start)

                    
                    # print('infer time', time.time() - start)
                metrics= self.evaluate(mode='test')
                print(metrics)
                self.log(metrics)
                if best_r_all < metrics["test/r_all"]:
                    waiting = 0
                    best_r_all = metrics["test/r_all"]
                    self.log({"best r_all": metrics["test/r_all"]})
                    print("best r all", best_r_all)
                elif epoch > self.config.min_epochs:
                    waiting += 1
                if waiting < self.patience:
                    self.train_loader = self.accelerator.prepare(
                        get_dataloader(
                            self.dataset["train"],
                            self.config.batch_size,
                            processor=self.processor,
                            mode="train",
                        )
                    )
                else:
                    break
        print("Finished Training")

    def evaluate(self, mode="val"):
        print("Evaluating current epoch", self.current_epoch)
        self.model.eval()
        loader = self.val_loader if mode == "val" else self.test_loader
        all_text_embeds = []
        all_vision_embeds = []
        with torch.no_grad():
            for data in tqdm(loader):
                text_embeds = self.model.get_text_features(
                    input_ids=data["input_ids"], attention_mask=data["attention_mask"]
                )
                vision_embeds = self.model.get_vision_features(
                    pixel_values=data["pixel_values"][0].unsqueeze(0)
                )
                all_text_embeds.append(text_embeds)
                all_vision_embeds.append(vision_embeds)

            all_text_embeds = torch.concat(all_text_embeds, 0)
            all_vision_embeds = torch.concat(all_vision_embeds, 0)
            if self.config.manifold == POINCARE:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds, device='cpu')
                sims_t2i = sims_t2i.detach().numpy()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            elif self.config.manifold == LORENTZ:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
                sims_t2i = sims_t2i.cpu().detach().numpy()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            else:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
                sims_t2i = sims_t2i.cpu().detach().numpy()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()

            metrics = evaluate_recall(sims_t2i=sims_t2i, mode=mode)
            # eu_metrics = evaluate_recall(sims_t2i=eu_sims_t2i, mode=mode)
            metrics["epoch"] = self.current_epoch
            # eu_metrics["epoch"] = self.current_epoch
        return metrics

    def save(self):
        torch.save(self.model, f"{self.save_dir}/{self.name}.pth")

    def load(self):
        self.model = torch.load(f"{self.save_dir}/{self.name}.pth")

class DistilTrainer:
    def __init__(
        self, config, model, dataset, train_loader, val_loader, test_loader, processor
    ):
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
        self.momentum = config.momentum
        self.processor = processor
        self.device = torch.device(
            f"cuda:{config.cuda}"
            if (torch.cuda.is_available() and config.cuda >= 0)
            else "cpu"
        )
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.enable_log = self.config.enable_log
        self.current_epoch = 0
        self.model = self.accelerator.prepare(model)
        self.dataset = dataset

        # if config.manifold == EUCLID:
        if config.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        # else:
        #     if config.optimizer == "adam":
        #         self.optimizer = RiemannianAdam(
        #             self.model.parameters(),
        #             lr=config.lr,
        #             stabilize=10,
        #             weight_decay=config.weight_decay,
        #         )
        #     else:
        #         self.optimizer = RiemannianSGD(
        #             self.model.parameters(),
        #             momentum=config.momentum,
        #             lr=config.lr,
        #             weight_decay=config.weight_decay,
        #             stabilize=10,
        #     )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.1, patience=2
        )
        
        (
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, train_loader, val_loader, test_loader, self.scheduler
        )
        self.name = f'{config.model_ckt.split("/")[-1]}_{config.manifold}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}_{config.batch_size}_{config.ft_out}'
        print("RUNNING:", self.name)

        if self.enable_log:
            wandb.init(name=self.name, config=vars(self.config), reinit=True, project="Graph")
        print("trainable parameters:", self.model.num_parameters())
        self.log({"trainable parameters": self.model.num_parameters()})

    def log(self, stat):
        if self.enable_log:
            wandb.log(stat)

    def train(self):
        # loop over the dataset multiple times
        best_r_all = 0.0
        waiting = 0
        for epoch in range(self.epochs):
            current_step = 0
            with self.accelerator.accumulate(self.model):
                self.model.train()
                self.current_epoch = epoch
                running_loss = 0.0
                print('train loader length:', len(self.train_loader))
                for data in tqdm(self.train_loader):
                    start = time.time()
                    self.accelerator.free_memory()
                    self.optimizer.zero_grad()
                    current_step += 1
                    # assert len(img_ids) == len(set(img_ids))

                    loss, stats = self.model(
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                        pixel_values=data["pixel_values"],
                        teacher_input_ids=data["input_ids"],
                        teacher_attention_mask=data["attention_mask"],
                        teacher_pixel_values=data["pixel_values"],
                    )

                    self.accelerator.backward(loss)
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()

                    running_loss += loss.item()

                    if (current_step + 1) % self.log_freq == 0:
                        self.log(stats)
                        print(stats)
                        print("Loss: {}".format(loss.item()))
                    if self.eval_freq != -1 and (current_step + 1) % self.eval_freq == 0:
                        metrics = self.evaluate(mode='val')
                        self.model.train()
                        print(metrics)
                        self.log(metrics)
                        self.scheduler.step(metrics["val/r_all"])
                    print('infer time', time.time() - start)

                    
                    # print('infer time', time.time() - start)
                metrics= self.evaluate(mode='test')
                print(metrics)
                self.log(metrics)
                if best_r_all < metrics["test/r_all"]:
                    waiting = 0
                    best_r_all = metrics["test/r_all"]
                    self.log({"best r_all": metrics["test/r_all"]})
                    print("best r all", best_r_all)
                elif epoch > self.config.min_epochs:
                    waiting += 1
                if waiting < self.patience:
                    self.train_loader = self.accelerator.prepare(
                        get_dataloader(
                            self.dataset["train"],
                            self.config.batch_size,
                            processor=self.processor,
                            mode="train",
                        )
                    )
                else:
                    break
        print("Finished Training")
    def get_itm_result(self, text_embeds:torch.Tensor, image_embeds:torch.Tensor, sims_t2i:torch.Tensor, k=50):
        indices = sims_t2i.topk(k).indices
        all_logits = []
        for i in range(indices.shape[0]):
            k_image_embeds = image_embeds.index_select(dim=0, index=indices[i].to(image_embeds.device))
            itm_text_inputs = text_embeds[i][0].expand(k_image_embeds.shape)
            itm_image_inputs = k_image_embeds 
            logits = self.model.itm_head(itm_image_inputs, itm_text_inputs).T
            print(indices[i])
            all_logits.append(torch.nn.functional.softmax(logits, dim=-1))
        
        all_logits = torch.cat(all_logits, dim=0)
        # print(all_logits.shape)
        # print((indices/5).int())
        return all_logits

            
        
        
        

    def evaluate(self, mode="val"):
        print("Evaluating current epoch", self.current_epoch)
        self.model.eval()
        loader = self.val_loader if mode == "val" else self.test_loader
        all_text_embeds = []
        all_vision_embeds = []
        img_ids = [] 
        text_ids = [] 
        with torch.no_grad():
            for data in tqdm(loader):

                text_embeds = self.model.get_text_features(
                    input_ids=data["input_ids"], attention_mask=data["attention_mask"]
                )
                vision_embeds = self.model.get_vision_features(
                    pixel_values=data["pixel_values"][0].unsqueeze(0)
                )
                all_text_embeds.append(text_embeds)
                all_vision_embeds.append(vision_embeds)
                img_ids.append(data['img_id'][0].unsqueeze_(0))
                text_ids.append(data['img_id'])

            all_text_embeds = torch.concat(all_text_embeds, 0)
            all_vision_embeds = torch.concat(all_vision_embeds, 0)
            text_ids = torch.concat(text_ids, 0).unsqueeze_(0)
            img_ids = torch.concat(img_ids, 0).unsqueeze_(0)
            # print(text_ids.shape)
            # print(img_ids.shape)
            targets = torch.eq(img_ids, text_ids.T).cpu().detach()
            if self.config.manifold == POINCARE:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds, device='cpu')
                sims_t2i = sims_t2i.detach()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            else: 
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
                sims_t2i = sims_t2i.cpu().detach()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            metrics = evaluate_recall(sims_t2i=sims_t2i, mode=mode)
            if self.config.use_itm_head:
                t2i_logits = self.get_itm_result(image_embeds=all_vision_embeds, text_embeds=all_text_embeds, sims_t2i=sims_t2i.to(self.device)) 
                print(t2i_logits)
                # itm_metrics = evaluate_recall(sims_t2i=t2i_logits, mode=mode)
            
            # eu_metrics = evaluate_recall(sims_t2i=eu_sims_t2i, mode=mode)
            metrics["epoch"] = self.current_epoch
            # eu_metrics["epoch"] = self.current_epoch
        return metrics

    def save(self):
        torch.save(self.model, f"{self.save_dir}/{self.name}.pth")

    def load(self):
        self.model = torch.load(f"{self.save_dir}/{self.name}.pth")
