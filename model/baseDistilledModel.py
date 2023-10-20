import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .modules.hyp_discriminator import HypDiscriminator 
from hyptorch.lorentz.manifold import CustomLorentz as Lorentz 
from hyptorch.geoopt.manifolds.lorentz import math as lmath 

from hyptorch.geoopt.manifolds.stereographic import PoincareBall 
from hyptorch.geoopt import Euclidean 
# from model.manifolds.lorentz import Lorentz 
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
import time


EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'


class BaseModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_ckt = config.model_ckt
        self.ft_out = config.ft_out
        self.clip_r = config.clip_radius
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.momentum = config.momentum
        self.queue_size = config.queue_size
        self.use_lorentz_centroid = config.use_lorentz_centroid

        manifold = config.manifold
    
        assert manifold in [EUCLID, POINCARE, LORENTZ]

        self.logit_scale = nn.Parameter(torch.tensor(1 / config.temp).log())
        self.weight_i2t = self.config.weight_i2t 
        self.curv = torch.as_tensor(config.curv if manifold != EUCLID else 0)
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())
        
    
        if manifold == EUCLID:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.clip_r = None
            self.manifold = Euclidean()
            self.discriminator = DisModel(dim=(256 if 'blip' in self.config.model_ckt else 512), layer_dims=[256, 1])
        elif manifold == POINCARE:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = PoincareBall(c=self.curv, learnable=config.curv_learnable)
            self.discriminator = HypDiscriminator(self.manifold, dim=(256 if 'blip' in self.config.model_ckt else 512), layer_dims=[512, 1])
        else: 
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable, atol=config.atol, rtol=config.rtol)
            self.discriminator = LorentzDisModel(self.manifold, dim=(256 if 'blip' in self.config.model_ckt else 512), layer_dims=[512])
        self.manifold_name =  manifold    
        self.vision_model = None 
        self.text_model = None 
        self.text_teacher = None
        self.vision_teacher = None

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params += sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        else:
            num_params += sum(p.numel() for p in self.text_model.parameters())
        return num_params

    def eval(self):
        self.vision_teacher.eval()
        self.text_teacher.eval()
        self.vision_model.body.eval()
        self.vision_model.head.eval()
        self.text_model.body.eval()
        self.text_model.head.eval()

    def train(self):
        self.vision_teacher.eval()
        self.text_teacher.eval()
        self.vision_model.body.train()
        self.vision_model.head.train()
        self.text_model.body.train()
        self.text_model.head.train()
        
    def dist_func(self, x, y, device='gpu'):
        if self.manifold_name == EUCLID:
            x = F.normalize(x,p=2, dim=-1) 
            y = F.normalize(y,p=2, dim=-1) 
            eu_dis = torch.matmul(x, y.t()) 
            return  eu_dis, eu_dis 
        elif self.manifold_name == POINCARE: 
            hyp_dist = -self.manifold.dist_batch(x, y, device=device)
            x = F.normalize(self.manifold.logmap0(x),p=2, dim=-1) 
            y = F.normalize(self.manifold.logmap0(y),p=2, dim=-1) 
            eu_dis = torch.matmul(x, y.t()) 
            return eu_dis, hyp_dist
        else: 
            hyp_dist = -self.manifold.dist_batch(x, y)
            x = F.normalize(self.manifold.logmap0(x),p=2, dim=-1) 
            y = F.normalize(self.manifold.logmap0(y),p=2, dim=-1) 
            eu_dis = torch.matmul(x, y.t()) 
            return eu_dis, hyp_dist
    
    def etailment_loss(self, text_feats:torch.Tensor, image_feats:torch.Tensor):
        entailment_loss = torch.tensor(0.0) 
        if isinstance(self.manifold, Lorentz):
            angle = self.manifold.oxy_angle(text_feats, image_feats)
            aperture = self.manifold.half_aperture(text_feats)
            entailment_loss = torch.clamp(angle - aperture, min=0).mean()
        
        return entailment_loss
        



    def itm_loss(self, imgs, cap, sims_i2t):
            
        bs = imgs.shape[0]
        weights_i2t = F.softmax(sims_i2t, dim=1)
        weights_t2i = F.softmax(sims_i2t.T, dim=1)
        mask = (torch.eye(bs) > 0).to(self.device)

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
        itm_labels = torch.cat(
            [
                torch.ones(bs,dtype=torch.float),
                torch.zeros(2*bs,dtype=torch.float)
            ],
        dim=0).view(-1,1).to(imgs.device)

        disc = self.discriminator(img_enc_all, cap_enc_all)
        class_weights = torch.tensor([[2.0]]).to(self.device)
        loss_itm = F.binary_cross_entropy_with_logits(disc, itm_labels, pos_weight=class_weights)
        itm_acc = ((disc >0.5).float() == itm_labels).float().sum()/itm_labels.shape[0]
        return loss_itm, itm_acc


    def margin_loss(self, sims_i2t, sims_i2i):
        bsize = sims_i2t.shape[0] 
        ones = torch.ones(bsize, bsize).to(self.device)
        pos_mask = torch.eye(bsize).to(self.device) 
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 
        neg_margin = self.config.euclid_img_neg_margin * neg_mask 
        pos_margin = self.config.euclid_pos_margin * pos_mask 
        sims_i2t = sims_i2t - neg_margin 
        sims_i2i = sims_i2i - neg_margin 
        sims_i2t = (sims_i2t - pos_margin) * sign 

        sims = torch.cat([torch.clamp(sims_i2t, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss

    def teacher_dist_func(self, x:torch.Tensor, y:torch.Tensor):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x, y.T) 
    
    def euclid_student_dist_func(self, x:torch.Tensor, y:torch.Tensor):
        if self.config.manifold == POINCARE:
            x = self.manifold.logmap0(x)
            y = self.manifold.logmap0(y)
        elif self.config.manifold == LORENTZ:
            x = self.manifold.get_space(x)
            y = self.manifold.get_space(y)
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x, y.T) 
            
        
    def itc_loss(self, image_embeds , text_embeds, image_embeds_t=None, text_embeds_t=None):
        # print(text_embeds_t.shape)
        # print(image_embeds_t.shape)

        bsize = text_embeds.shape[0]
        eye_mask = torch.eye(bsize).to(self.device) * 1e9
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()
        target = torch.arange(bsize).to(self.device)
        eu_sims_i2t, sims_i2t = self.dist_func(image_embeds, text_embeds) 
        eu_sims_t2i, sims_t2i = eu_sims_i2t.T, sims_i2t.T
        eu_sims_i2i, sims_i2i = self.dist_func(image_embeds, image_embeds) 
        eu_sims_t2t, sims_t2t = self.dist_func(text_embeds, text_embeds) 
        if image_embeds_t is not None and text_embeds_t is not None:
            teacher_sims_i2t = self.teacher_dist_func(image_embeds_t, text_embeds_t) 
            teacher_sims_t2i = teacher_sims_i2t.T
            student_sims_i2t = self.euclid_student_dist_func(image_embeds, text_embeds) 
            studen_sims_t2i = teacher_sims_i2t.T
            soft_targets_i2t = nn.functional.softmax(teacher_sims_i2t*_scale, dim=-1)
            soft_targets_t2i = nn.functional.softmax(teacher_sims_t2i*_scale, dim=-1)
            soft_prob_i2t = nn.functional.log_softmax(student_sims_i2t*_scale, dim=-1)
            soft_prob_t2i = nn.functional.log_softmax(studen_sims_t2i*_scale, dim=-1)
            soft_loss_i2t = -torch.sum(
                soft_targets_i2t * soft_prob_i2t, dim=-1
            ).mean()
            soft_loss_t2i = -torch.sum(
                soft_targets_t2i * soft_prob_t2i, dim=-1
            ).mean()      
            soft_itc_loss = self.weight_i2t * soft_loss_i2t + (1 - self.weight_i2t) * soft_loss_t2i 
        else: 
            soft_itc_loss = torch.tensor(0.0)

        logits_i2t = torch.cat([sims_i2t, sims_i2i-eye_mask], dim=1) * _scale
        logits_t2i = torch.cat([sims_t2i, sims_t2t-eye_mask], dim=1) * _scale
        
        
        margin_loss = torch.tensor(0.0)
        # if self.config.use_margin_loss:
            # margin_loss = 0.5 * (self.margin_loss(eu_sims_i2t, eu_sims_t2t - eye_mask) + self.margin_loss(eu_sims_t2i, eu_sims_i2i - eye_mask))
        
        itc_loss =  self.weight_i2t * F.cross_entropy(logits_i2t, target) + (1 - self.weight_i2t) * F.cross_entropy(logits_t2i, target) 
        
        loss = (1 - self.config.soft_target_loss) * itc_loss + self.config.soft_target_loss * soft_itc_loss  + margin_loss
        
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/margin_loss": margin_loss.item(),
            "logits/itc_loss": itc_loss.item(),
            "logits/soft_itc_loss": soft_itc_loss.item(),
            "logits/min": sims_i2t.min().item(),
            "logits/mean": sims_i2t.mean().item(),
            "logits/max": sims_i2t.max().item(),
            "logits/acc": (sims_i2t.argmax(-1) == target).float().mean().item(),
            "logits/eu_acc": (eu_sims_i2t.argmax(-1) == target).float().mean().item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }
        return loss, stats, sims_i2t

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_input_ids: Optional[torch.LongTensor] = None,
        teacher_pixel_values: Optional[torch.FloatTensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CLIPOutput]:
        with torch.no_grad():
            teacher_vision_outputs = self.vision_teacher(
                teacher_pixel_values
            )
            teacher_text_outputs = self.text_teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )
            image_embeds_t = teacher_vision_outputs[1]
            text_embeds_t = teacher_text_outputs[1]


        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        self.manifold.assert_check_point_on_manifold(image_embeds)
        self.manifold.assert_check_point_on_manifold(text_embeds)

        itc_loss, stats, sims_i2t = self.itc_loss(
            image_embeds=image_embeds, 
            text_embeds=text_embeds, 
            image_embeds_t=image_embeds_t, 
            text_embeds_t=text_embeds_t
        )

        itm_loss, itm_acc = self.itm_loss(image_embeds, text_embeds, sims_i2t=sims_i2t)
        stats["logits/itm_loss"] = itm_loss.item() 
        stats["logits/itm_acc"] = itm_acc.item() 
        loss = itm_loss + itc_loss 
        return loss, stats

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return text_outputs[1]

    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        return vision_outputs[1] 


