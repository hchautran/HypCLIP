from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef_models import compute_sim_matrix
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
)
from lavis.models.blip_models.blip import BlipBase
from torch import nn

from hyptorch.lorentz.manifold import CustomLorentz as Lorentz 
from hyptorch.geoopt import Euclidean, PoincareBall 
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .modules.utils import (
    prepare_encoder,
    ManifoldMapper
) 
from .modules.fuseModel import FuseEncoder

EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'
class Text(object):
    pass


class BaseModelWithQueue(BlipBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    BLIP retrieval model.

    Supported model types:
        - coco: fine-tuned BLIP base model on COCO dataset (Karpathy split).
        - flickr: fine-tuned BLIP base model on Flickr30k dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_retrieval", "coco")
        >>> model = load_model("blip_retrieval", "flickr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/blip_retrieval_coco.yaml",
        "flickr": "configs/models/blip_retrieval_flickr.yaml",
    }

    def __init__(
        self,
        config,
        models,
    ):
        """ """
        super().__init__()
        self.config = config
        self.models=models 
        self.num_models = len(models)
        self.clip_r = config.clip_radius
        self.queue_size = config.queue_size
        self.weight_i2t = config.weight_i2t
        self.weight_retrieval = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        assert config.manifold in [EUCLID, POINCARE, LORENTZ]
        self.mapper = None           
        self.curv = torch.as_tensor(config.curv if config.manifold != EUCLID else 0)
        class_weight = torch.tensor([0.5, 1.0])
        self.itm_criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())


        if config.manifold == EUCLID:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.clip_r = None
            self.manifold = Euclidean()
        elif config.manifold == POINCARE:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = PoincareBall(c=self.curv, learnable=config.curv_learnable)
            self.mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
        else: 
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable, atol=config.atol, rtol=config.rtol)
            self.mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.model_m= None 
        vis_encoders, text_encoders, text_head, vision_head, d_visions, d_texts = prepare_encoder(config, models)
        self.model = FuseEncoder(
            config,           
            d_visions=d_visions, 
            d_texts=d_texts, 
            ft_out=config.ft_out, 
            vision_bodies=vis_encoders, 
            text_bodies=text_encoders,
            vision_head=vision_head,
            text_head=text_head,
            mapper=(self.mapper if config.manifold != EUCLID else None),
            manifold=(self.manifold if config.manifold != EUCLID else None), 
        )

        self.momentum = config.momentum
        self.logit_scale = nn.Parameter(torch.tensor(config.temp))

        self.alpha = config.alpha
        self.max_txt_len = config.max_txt_len
        self._init_queue(config, config.ft_out)
    
    def _init_queue(self, config, ft_out):
        self.model_m= deepcopy(self.model) 
        self.model_pairs = [
            [self.model, self.model_m],
        ]
        self.copy_params()
          # create the queue
        if config.manifold == EUCLID:
            self.register_buffer("image_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("text_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("image_ori_queue", torch.randn(self.queue_size, 256).T)
            self.register_buffer("text_ori_queue", torch.randn(self.queue_size, 256).T)
            self.image_queue = F.normalize(self.image_queue.T, dim=-1).T
            self.text_queue = F.normalize(self.text_queue.T, dim=-1).T
            self.image_ori_queue = F.normalize(self.image_ori_queue.T, dim=-1).T
            self.text_ori_queue = F.normalize(self.text_ori_queue.T, dim=-1).T
        elif config.manifold == POINCARE:
            self.register_buffer("image_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("text_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("image_ori_queue", torch.randn(self.queue_size, 256).T)
            self.register_buffer("text_ori_queue", torch.randn(self.queue_size, 256).T)
            self.image_queue = F.normalize(self.image_queue.T, dim=-1).T
            self.text_queue = F.normalize(self.text_queue.T, dim=-1).T
            self.image_ori_queue = F.normalize(self.image_ori_queue.T, dim=-1).T
            self.text_ori_queue = F.normalize(self.text_ori_queue.T, dim=-1).T
            self.image_queue = self.manifold.expmap0(self.image_queue.T, dim=-1).T
            self.text_queue = self.manifold.expmap0(self.text_queue.T, dim=-1).T
            self.image_ori_queue = self.manifold.expmap0(self.image_ori_queue.T, dim=-1).T
            self.text_ori_queue = self.manifold.expmap0(self.text_ori_queue.T, dim=-1).T
        else:
            self.register_buffer("image_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.register_buffer("text_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.register_buffer("image_ori_queue", self.manifold.random(self.queue_size, 257).T)
            self.register_buffer("text_ori_queue", self.manifold.random(self.queue_size, 257).T)

        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))
    
    
    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
            
    def get_euclid_dist(self, x:torch.Tensor, y:torch.tensor):
        if self.config.manifold == LORENTZ:
            x = self.manifold.get_space(x)
            y = self.manifold.get_space(y)
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x, y.T) 
            
    def dist_func(self, x:torch.Tensor, y:torch.Tensor): 
        if self.config.manifold == EUCLID:
            eu_dis = torch.matmul(x, y.T) 
            return  eu_dis 
        elif self.config.manifold == POINCARE: 
            hyp_dist = -self.manifold.dist_batch(x, y, device=f'{"gpu" if self.training else "cpu"}')
            return hyp_dist
        else: 
            hyp_dist = -self.manifold.dist_batch(x, y)
            return hyp_dist
    def eu_margin_loss(self, pos_mask, sims, sims_i2i):
        ones = torch.ones_like(pos_mask).to(self.device)
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 

        neg_margin = self.config.euclid_neg_margin * neg_mask 
        pos_margin = self.config.euclid_pos_margin * pos_mask 
        sims = sims - neg_margin 
        sims_i2i = sims_i2i - neg_margin 
        sims = (sims - pos_margin) * sign 

        sims = torch.cat([torch.clamp(sims, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss

    def hyp_margin_loss(self, pos_mask, sims, sims_i2i):
        ones = torch.ones_like(pos_mask).to(self.device)
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 
        neg_margin = self.config.lorentz_neg_margin * neg_mask 
        pos_margin = self.config.lorentz_pos_margin * pos_mask 
        sims = sims + neg_margin 
        sims_i2i = sims_i2i + neg_margin 
        sims = (sims + pos_margin) * sign 

        sims = torch.cat([torch.clamp(sims, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss
    
    def margin_loss(self, pos_idx, text_feat, image_feat, text_world, image_world):
        if not self.config.use_margin_loss:
            return torch.tensor(0.0)

        sim_i2i = self.dist_func(image_feat, image_world) 
        sim_t2t = self.dist_func(text_feat, text_world) 
        if self.config.manifold ==  LORENTZ:
            return self.hyp_margin_loss(pos_idx, sim_i2i, sim_t2t) 
        return self.eu_margin_loss(pos_idx, sim_i2i, sim_t2t) 

    
    def postprocess_embeds(self, embed):
        if self.mapper is not None:
            self.manifold.assert_check_point_on_manifold(embed)
            return embed 
        else:
            return F.normalize(embed, p=2, dim=-1) 


    def itm_loss(self, idx, text_hidden_states, image_hidden_states, sim_t2i, sim_i2t):
        bs = text_hidden_states.shape[0]
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())
            
            weights_t2i = F.softmax(sim_t2i/self.logit_scale, dim=1) + 1e-4
            weights_i2t = F.softmax(sim_i2t/self.logit_scale, dim=1) + 1e-4
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_hidden_states[neg_idx])

        # select a negative text for each image
        text_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_hidden_states[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        text_hidden_states = torch.cat(
            [text_hidden_states, text_hidden_states, text_ids_neg], dim=0
        )  # pos, pos, neg

        image_hidden_states = torch.cat(
            [image_hidden_states, image_embeds_neg, image_hidden_states], dim=0
        )  # pos, neg, pos


        itm_score = self.model.compute_itm(
            text_latents=text_hidden_states, 
            vision_latents=image_hidden_states, 
        ) 

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        itm_acc = (itm_score.argmax(-1) == itm_labels).float().sum()/itm_labels.shape[0]

        loss_itm = self.itm_criterion(itm_score, itm_labels) 
        return loss_itm,  itm_acc


    def forward(
        self, 
        data,
        epoch: int,
        iters: int,
        num_iters_per_epoch:int,
    ):

        idx = data['img_id'] 
        input_ids = []
        attention_masks = []
        pixel_values = []
        for i in range(self.num_models):
            input_ids.append(data[f'input_ids_{i}'])
            pixel_values.append(data[f'pixel_values_{i}'])
            attention_masks.append(data[f'attention_mask_{i}'])

        alpha = self.alpha * self._rampup_factor(
            epoch=epoch,
            iters=iters,
            num_iters_per_epoch=num_iters_per_epoch,
        )

        text_output  = self.model(
            input_ids=input_ids,
            attention_masks=attention_masks
        )
        image_output = self.model(
            pixel_values=pixel_values
        )
        text_embeds = text_output[1] 
        image_embeds = image_output[1] 
        text_embeds_ori = text_output[2] 
        image_embeds_ori = image_output[2] 
        text_hidden_states = text_output[0]
        vision_hidden_states = image_output[0]

        text_feat = self.postprocess_embeds(text_embeds)
        image_feat = self.postprocess_embeds(image_embeds)
        image_feat_ori = self.postprocess_embeds(image_embeds_ori)
        text_feat_ori = self.postprocess_embeds(text_embeds_ori)

        self.manifold.assert_check_point_on_manifold(text_feat)
        self.manifold.assert_check_point_on_manifold(image_feat)
        self.manifold.assert_check_point_on_manifold(text_feat_ori)
        self.manifold.assert_check_point_on_manifold(image_feat_ori)
        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self.logit_scale.clamp_(0.001, 0.5)
            self._momentum_update()
            image_output_m = self.model_m(
                pixel_values=pixel_values 
            )

            text_output_m = self.model_m(
                input_ids=input_ids,
                attention_masks=attention_masks,
            )

            image_feat_m = self.postprocess_embeds(image_output_m[1])
            text_feat_m = self.postprocess_embeds(text_output_m[1])

            image_feat_world= torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_world= torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            image_feat_ori_world= torch.cat([image_feat_ori.t(), self.image_ori_queue.clone().detach()], dim=1)
            text_feat_ori_world = torch.cat([text_feat_ori.t(), self.text_ori_queue.clone().detach()], dim=1)

            sim_i2t_ori = self.dist_func(image_feat_ori, text_feat_ori_world.T) 
            sim_t2i_ori = self.dist_func(text_feat_ori, image_feat_ori_world.T)
            ori_i2t_target = F.softmax(sim_i2t_ori / self.logit_scale, dim=-1)
            ori_t2i_target = F.softmax(sim_t2i_ori / self.logit_scale, dim=-1)


            self.manifold.assert_check_point_on_manifold(text_feat_world.T)
            self.manifold.assert_check_point_on_manifold(image_feat_world.T)

        sim_i2t= self.dist_func(image_feat, text_feat_world.T) 
        sim_t2i= self.dist_func(text_feat, image_feat_world.T)

        fused_loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t / self.logit_scale, dim=1) * ori_i2t_target, dim=-1
        ).mean()
        fused_loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i / self.logit_scale, dim=1) * ori_t2i_target, dim=-1
        ).mean()      
        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t / self.logit_scale, dim=1) * sim_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i / self.logit_scale, dim=1) * sim_targets, dim=-1
        ).mean()      
        margin_loss  = self.margin_loss(pos_idx=pos_idx, text_feat=text_feat, image_feat=image_feat, text_world=text_feat_world.T, image_world=image_feat_world.T)
    
        fused_loss_itc = self.config.weight_i2t * fused_loss_i2t + (1-self.config.weight_i2t) * fused_loss_t2i
        loss_itc = self.config.weight_i2t * loss_i2t + (1-self.config.weight_i2t) * loss_t2i
      
        sims = self.dist_func(image_feat, text_feat)
        loss_itm, itm_acc = self.itm_loss(
            idx=idx,
            text_hidden_states=text_hidden_states, 
            image_hidden_states=vision_hidden_states, 
            sim_i2t=sims, 
            sim_t2i=sims.T
        )
        loss =  (1-alpha)*fused_loss_itc+ loss_itc + loss_itm + margin_loss

        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/margin_loss": margin_loss.item(),
            "logits/itm_loss": loss_itm.item(),
            "logits/min": sims.min().item(),
            "logits/mean": sims.mean().item(),
            "logits/max": sims.max().item(),
            "logits/alpha": alpha,
            "logits/acc": (sims.argmax(-1) == torch.arange(bsize).to(self.device)).float().mean().item(),
            "logits/temp": self.logit_scale.item(),
            "logits/itm_acc": itm_acc.item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, image_feat_ori, text_feat_ori, idx)
        return  loss, stats

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
    
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_ori, text_ori, idxs=None):
        # gather keys before updating queue
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        self.image_ori_queue[:, ptr : ptr + batch_size] = image_ori.T
        self.text_ori_queue[:, ptr : ptr + batch_size] = text_ori.T

        if idxs is not None:
            idxs = idxs
            self.idx_queue[:, ptr : ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr


    def get_text_features(
        self,
        data
    ):
        input_ids = []
        attention_masks = []
        for i in range(self.num_models):
            input_ids.append(data[f'input_ids_{i}'].squeeze())
            attention_masks.append(data[f'attention_mask_{i}'].squeeze())

        text_output = self.model(
           input_ids=input_ids, 
           attention_masks=attention_masks, 
        )
        text_feat = self.postprocess_embeds(text_output[1])
        text_ori= self.postprocess_embeds(text_output[2])
        return text_feat, text_output[0], text_ori

    def get_vision_features(self, data):
        pixel_values = []
        for i in range(self.num_models):
            pixel_values.append(data[f'pixel_values_{i}'].squeeze(0))
        image_output = self.model(pixel_values=pixel_values)
        image_feat = self.postprocess_embeds(image_output[1])
        image_ori = self.postprocess_embeds(image_output[2])
        return image_feat, image_output[0], image_ori
    