import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .manifolds.euclidean import Euclidean 
from hyptorch.lorentz.manifold import CustomLorentz as Lorentz 
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
)


EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'


class BaseModel(nn.Module, MomentumDistilationMixin, SharedQueueMixin):
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

        self.temp = nn.Parameter(torch.as_tensor(config.temp), requires_grad=config.temp != 0) 
        self.weight_i2t = nn.Parameter(torch.as_tensor(0.5), requires_grad=False) 

        self.curv = torch.as_tensor(config.curv if manifold != EUCLID else 0)
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())
        
        self.visual_encoder_m = None
        self.text_encoder_m = None
        self.model_pairs = None 
    
        if manifold == EUCLID:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.clip_r = None
            self.manifold = Euclidean()
            self.discriminator = DisModel(dim=config.ft_out, layer_dims=[256, 256, 256, 1])
        else: 
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable, atol=config.atol, rtol=config.rtol)
            self.discriminator = LorentzDisModel(self.manifold, dim=config.ft_out, layer_dims=[256, 256,256])
        self.manifold_name =  manifold    
        self.vision_model = None 
        self.text_model = None 

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params += sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params += sum(p.numel() for p in self.parameters())
        return num_params

    def eval(self):
        self.vision_model.body.eval()
        self.vision_model.head.eval()
        self.text_model.body.eval()
        self.text_model.head.eval()

    def train(self):
        self.vision_model.body.train()
        self.vision_model.head.train()
        self.text_model.body.train()
        self.text_model.head.train()
        
    def dist_func(self, x, y):
        if self.manifold_name == EUCLID:
            # print('calulating dot product')
            x = F.normalize(x,p=2, dim=-1) 
            y = F.normalize(y,p=2, dim=-1) 
            return torch.matmul(x, y.t()) 
        else: 
            # print('calculating lorentz distance')
            return -self.manifold.sqdist_batch(x, y)


    def itm_loss(self, imgs, cap, sims_i2t):
        bs = imgs.shape[0]
        weights_i2t = F.softmax(sims_i2t, dim=1)
        weights_t2i = F.softmax(sims_i2t.T, dim=1)
        mask = (torch.eye(bs)>1).to(self.device)

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
        print(disc.shape)
        loss_itm = F.binary_cross_entropy_with_logits(disc, itm_labels)
        return loss_itm


    def contrastive_loss(self, sims_i2t, sims_i2i):
        bsize = sims_i2t.shape[0] 
        ones = torch.ones(bsize, bsize).to(self.device)
        pos_mask = torch.eye(bsize).to(self.device) 
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 
        if self.config.manifold == EUCLID:
            neg_margin = self.config.euclid_neg_margin * neg_mask 
            pos_margin = self.config.euclid_pos_margin * pos_mask 
            sims_i2t = sims_i2t - neg_margin 
            sims_i2i = sims_i2i - neg_margin 
            sims_i2t = (sims_i2t - pos_margin) * sign 
        else:
            neg_margin = self.config.lorentz_neg_margin * neg_mask 
            pos_margin = self.config.lorentz_pos_margin * pos_mask 
            sims_i2t = sims_i2t + neg_margin 
            sims_i2i = sims_i2i + neg_margin 
            sims_i2t = (sims_i2t + pos_margin) * sign 

        sims = torch.cat([torch.clamp(sims_i2t, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss
        
    def itc_loss(self, image_embeds , text_embeds):
        bsize = text_embeds.shape[0]
        eye_mask = torch.eye(bsize).to(self.device) * 1e9
        sims_i2t = self.dist_func(image_embeds, text_embeds)
        sims_t2i = sims_i2t.T
        sims_i2i = self.dist_func(image_embeds, image_embeds)
        sims_t2t = self.dist_func(text_embeds, text_embeds)
        target = torch.arange(bsize).to(self.device)
        logits_i2t = torch.cat([sims_i2t/self.temp, sims_i2i/self.temp - eye_mask], dim=1)
        logits_t2i = torch.cat([sims_t2i/self.temp, sims_t2t/self.temp - eye_mask], dim=1)

        contrastive_loss = self.contrastive_loss(sims_i2t, sims_i2i - eye_mask) + self.contrastive_loss(sims_t2i, sims_t2t - eye_mask)
        itc_loss =  F.cross_entropy(logits_i2t, target) + F.cross_entropy(logits_t2i, target) 
        loss = itc_loss + contrastive_loss 
        
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t.item(),
            "logits/contrastive_loss": contrastive_loss.item() if contrastive_loss is not None else 0.0,
            "logits/itc_loss": itc_loss.item(),
            "logits/min": sims_i2t.min().item(),
            "logits/mean": sims_i2t.mean().item(),
            "logits/max": sims_i2t.max().item(),
            "logits/acc": (sims_i2t.argmax(-1) == target).float().mean().item(),
        }
        return loss, stats, sims_i2t

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        itm_loss = self.itm_loss(image_embeds, text_embeds, sims_i2t=sims_i2t)
        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itm_loss + itc_loss
        return loss, stats, itc_loss, itm_loss

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_embeds = text_outputs[1]
        return text_embeds

    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        return vision_outputs[1] 


