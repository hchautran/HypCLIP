import wandb
from accelerate import Accelerator
from utils.data_utils import get_dataloader
from hyptorch.geoopt.optim import RiemannianAdam, RiemannianSGD
from utils.retrivial_utils import report_metrics 
from tqdm.auto import tqdm
import torch
from config import EUCLID, POINCARE, LORENTZ
from config import CLIP_BASE_PATCH_16, CLIP_BASE_PATCH_32, CLIP_LARGE_PATCH_14, BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO
import time

names = {
   CLIP_BASE_PATCH_32: 'clip_base_32', 
   CLIP_BASE_PATCH_16: 'clip_base_16',
   CLIP_LARGE_PATCH_14: 'clip_large_14', 
   BLIP_BASE_FLICKR: 'hf_blip_base', 
   LAVIS_BLIP_BASE_FLICKR: 'lv_blip_base_FLICKR',
   LAVIS_BLIP_BASE_COCO: 'lv_blip_base_COCO' 
}

class MyTrainer:
    def __init__(
        self, config, model, train_loader, val_loader, test_loader
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
            self.optimizer, 'max', factor=0.1, patience=1
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
        self.name = f'{names[config.model_ckt]}_{config.manifold}_{config.vision_trainable_blocks}_{config.text_trainable_blocks}_{config.batch_size}_{config.use_graph}'
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
                    if data['pixel_values'].shape[0] < self.config.batch_size: break
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

                        test_metrics = self.evaluate(mode='test')
                        val_metrics = self.evaluate(mode='val')
                        self.log(test_metrics)
                        self.log(val_metrics)
                        print(test_metrics)
                        print(val_metrics)

                        self.scheduler.step(test_metrics["test/r_all"])
                        if best_r_all < test_metrics["test/r_all"]:
                            best_r_all = test_metrics["test/r_all"]
                            self.log({"best r_all": test_metrics["test/r_all"]})
                            print("best r all", best_r_all)

                        self.model.train()

                    
        print("Finished Training")

    def get_itm_result(self, text_embeds:torch.Tensor, image_embeds:torch.Tensor, sims_t2i:torch.Tensor, k=10):
        indices = sims_t2i.topk(k).indices
        all_logits = []
        for i in range(indices.shape[0]):
            k_image_embeds = image_embeds.index_select(dim=0, index=indices[i].to(image_embeds.device))
            itm_text_inputs = text_embeds[i].expand(k_image_embeds.shape)
            itm_image_inputs = k_image_embeds 
            logits = self.model.itm_head(itm_image_inputs, itm_text_inputs).T
            all_logits.append(torch.nn.functional.softmax(logits, dim=-1))
        
        all_logits = torch.cat(all_logits, dim=0)
        return all_logits, indices

            
            

    def evaluate(self, mode="val"):
        from torch.utils.data import DataLoader
        print("Evaluating current epoch", self.current_epoch)
        self.model.eval()

        dataset = self.val_loader if mode == "val" else self.test_loader
        all_text_embeds = []
        all_vision_embeds = []

        loader = self.accelerator.prepare(DataLoader(dataset, batch_size=1, shuffle=False))
        with torch.no_grad():
            for data in tqdm(loader):
                text_embeds = self.model.get_text_features(
                    input_ids=data["input_ids"][0], attention_mask=data["attention_mask"][0]
                )
                vision_embeds = self.model.get_vision_features(
                    pixel_values=data["pixel_values"][0]
                )
                all_text_embeds.append(text_embeds)
                all_vision_embeds.append(vision_embeds)

            all_text_embeds = torch.concat(all_text_embeds, 0)
            all_vision_embeds = torch.concat(all_vision_embeds, 0)
            if self.config.manifold == POINCARE:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds, device='cpu')
                sims_t2i = sims_t2i.detach()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            elif self.config.manifold == LORENTZ:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
                sims_t2i = sims_t2i.cpu().detach()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()
            else:
                sims_t2i = self.model.dist_func(all_text_embeds, all_vision_embeds)
                sims_t2i = sims_t2i.cpu().detach()
                # eu_sims_t2i = eu_sims_t2i.cpu().detach().numpy()

            metrics = report_metrics(scores_t2i=sims_t2i, scores_i2t=sims_t2i.T, img2txt=dataset.img2txt, txt2img=dataset.txt2img, mode=mode )
            # eu_metrics = evaluate_recall(sims_t2i=eu_sims_t2i, mode=mode)
            metrics["epoch"] = self.current_epoch

        return metrics

    def save(self):
        torch.save(self.model, f"{self.save_dir}/{self.name}.pth")

    def load(self):
        self.model = torch.load(f"{self.save_dir}/{self.name}.pth")

