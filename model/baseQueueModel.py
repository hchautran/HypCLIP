from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef_models import compute_sim_matrix
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather,
)
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn

from hyptorch.lorentz.manifold import CustomLorentz as Lorentz 
from hyptorch.geoopt import PoincareBall 
from hyptorch.geoopt import Euclidean 
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .modules.hyp_discriminator import HypDiscriminator 
from .modules.utils import ManifoldMapper
from peft import get_peft_model, LoraConfig, TaskType

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
    ):
        """ """
        super().__init__()
        self.config = config
        self.model_ckt = config.model_ckt
        self.clip_r = config.clip_radius
        self.queue_size = config.queue_size
        self.vision_model = None
        self.text_model = None
        self.weight_i2t = config.weight_i2t
        assert config.manifold in [EUCLID, POINCARE, LORENTZ]
        self.mapper = None           
        self.curv = torch.as_tensor(config.curv if config.manifold != EUCLID else 0)
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())


        if config.manifold == EUCLID:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.clip_r = None
            self.manifold = Euclidean()
            # self.itm_head = DisModel(dim=self.ft_out, layer_dims=[256, 1])
        elif config.manifold == POINCARE:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = PoincareBall(c=self.curv, learnable=config.curv_learnable)
            # self.itm_head = HypDiscriminator(self.manifold, dim=self.ft_out, layer_dims=[256, 1])
            self.mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
        else: 
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable, atol=config.atol, rtol=config.rtol)
            # self.itm_head = LorentzDisModel(self.manifold, dim=self.ft_out, layer_dims=[256])
            self.mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)

        # create the momentum encoder
        # TODO
        self.vision_model_m= None 
        self.text_model_m = None 

        self.momentum = config.momentum
        self.logit_scale = nn.Parameter(torch.tensor(1 / config.temp).log())

        self.alpha = config.alpha
        self.max_txt_len = config.max_txt_len
        self.negative_all_rank = config.negative_all_rank

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))
    
    
    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    
    def eval(self):
        NotImplementedError


    def train(self):
        NotImplementedError
  
    
    def get_euclid_dist(self, x:torch.Tensor, y:torch.tensor):
        if self.config.manifold == LORENTZ:
            x = self.manifold.get_space(x)
            y = self.manifold.get_space(y.T).T
        if self.config.manifold == POINCARE:
            x = self.manifold.logmap0(x)
            y = self.manifold.logmap0(y.T).T
        x = F.normalize(x, p=2, dim=-1) 
        y = F.normalize(y, p=2, dim=-1) 
        return torch.matmul(x, y) 
            
    def dist_func(self, x:torch.Tensor, y:torch.Tensor, device='gpu'):
        if self.config.manifold == EUCLID:
            x = F.normalize(x,p=2, dim=-1) 
            y = F.normalize(y,p=2, dim=-1) 
            eu_dis = torch.matmul(x, y) 
            return  eu_dis 
        elif self.config.manifold == POINCARE: 
            hyp_dist = -self.manifold.dist_batch(x, y.T, device=device)

            return hyp_dist
        else: 
            hyp_dist = -self.manifold.dist_batch(x, y.T)
            return hyp_dist
    
    def postprocess_embeds(self, embed):
        if self.mapper is not None:
            self.manifold.assert_check_point_on_manifold(embed)
            return embed 
        else:
            return F.normalize(embed, p=2, dim=-1) 
    
    def eu_margin_loss(self, pos_mask,sims, sims_i2i):
        ones = torch.ones_like(pos_mask).to(self.device)
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 
        # if self.config.manifold == EUCLID:
        neg_margin = self.config.euclid_img_neg_margin * neg_mask 
        pos_margin = self.config.euclid_pos_margin * pos_mask 
        sims = sims - neg_margin 
        sims_i2i = sims_i2i - neg_margin 
        sims = (sims - pos_margin) * sign 
        sims = torch.clamp(sims, min=0.0)
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
        sims = torch.clamp(sims, min=0.0)
        sims = torch.cat([torch.clamp(sims, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss
        

    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_id: torch.FloatTensor,
        epoch: int,
        iters: int,
        num_iters_per_epoch:int,
    ):
        idx = image_id

        alpha = self.alpha * self._rampup_factor(
            epoch=epoch,
            iters=iters,
            num_iters_per_epoch=num_iters_per_epoch,
        )

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()
        
        # TODO 
        
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_output = self.vision_model(
            pixel_values=pixel_values
        )
        text_embeds = text_output[1] 
        image_embeds = image_output[1] 

        text_feat = self.postprocess_embeds(text_embeds)
        image_feat = self.postprocess_embeds(image_embeds)
        self.manifold.assert_check_point_on_manifold(text_feat)
        self.manifold.assert_check_point_on_manifold(image_feat)
        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        pos_mask = pos_idx * 1e9
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)


        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.vision_model_m(
                pixel_values=pixel_values 
            )
            image_feat_m = self.postprocess_embeds(image_embeds_m[1])
            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_embeds_m = self.text_model_m(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_feat_m = self.postprocess_embeds(text_embeds_m[1])
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = self.dist_func(image_feat_m, text_feat_m_all) * _scale 
            sim_t2i_m = self.dist_func(text_feat_m, image_feat_m_all) * _scale 


            self.manifold.assert_check_point_on_manifold(text_feat_m_all.T)
            self.manifold.assert_check_point_on_manifold(image_feat_m_all.T)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets


        sim_i2t = self.dist_func(image_feat, text_feat_m_all) 
        sim_t2i = self.dist_func(text_feat, image_feat_m_all) 
        sim_i2i = self.dist_func(image_feat, image_feat_m_all) - pos_mask 
        sim_t2t = self.dist_func(text_feat, text_feat_m_all) - pos_mask 

        margin_loss = torch.tensor(0.0)


        if self.config.use_margin_loss and self.config.manifold == EUCLID:
            margin_loss = (self.eu_margin_loss(idx, sim_i2t, sim_t2t) + self.eu_margin_loss(idx, sim_t2i, sim_i2i)) / 2
        elif self.config.use_margin_loss and self.config.manifold != EUCLID:
            eu_sim_i2t = self.get_euclid_dist(image_feat, text_feat_m_all) 
            eu_sim_t2i = self.get_euclid_dist(text_feat, image_feat_m_all) 
            eu_sim_i2i = self.get_euclid_dist(image_feat, image_feat_m_all) - pos_mask 
            eu_sim_t2t = self.get_euclid_dist(text_feat, text_feat_m_all) - pos_mask 
            if self.config.hyp_margin_loss_weight > 0.0:
                eu_margin_loss = (self.eu_margin_loss(pos_idx, eu_sim_i2t, eu_sim_t2t) + self.eu_margin_loss(pos_idx, eu_sim_t2i, eu_sim_i2i)) / 2
                hyp_margin_loss = (self.hyp_margin_loss(pos_idx, sim_i2t, sim_t2t) + self.hyp_margin_loss(pos_idx, sim_t2i, sim_i2i)) / 2
                margin_loss = self.config.hyp_margin_loss_weight * hyp_margin_loss +  eu_margin_loss
            else:
                margin_loss = (self.eu_margin_loss(pos_idx, eu_sim_i2t, eu_sim_t2t) + self.eu_margin_loss(pos_idx, eu_sim_t2i, eu_sim_i2i)) / 2
            
        
           

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t * _scale, dim=1) * sim_i2t_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i * _scale, dim=1) * sim_t2i_targets, dim=-1
        ).mean()      

        loss_itc = (loss_i2t + loss_t2i) / 2

        in_batch_target = torch.arange(bsize).to(self.device)
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/margin_loss": margin_loss.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc": (self.dist_func(image_feat, text_feat.T).argmax(-1) == in_batch_target).float().mean().item(),
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        return stats, BlipOutput(
            loss=loss_itc + margin_loss,
            loss_itc=loss_itc,
            # loss_itm=loss_itm,
            sims=BlipSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                image_embeds_m=image_embeds_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
            ),
        )

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        position_ids = None
    ):
        # TODO 
        text_output = self.text_model(
           input_ids=input_ids, 
           attention_mask=attention_mask, 
        )
        text_feat = self.postprocess_embeds(text_output[1])
        return text_feat

    def get_vision_features(self, pixel_values:torch.Tensor):
        # TODO 
        image_output = self.vision_model(pixel_values=pixel_values)
        image_feat = self.postprocess_embeds(image_output[1])
        return image_feat
