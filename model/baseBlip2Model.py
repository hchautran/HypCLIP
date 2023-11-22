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
        self.mapper = None

        manifold = config.manifold
    
        assert manifold in [EUCLID, POINCARE, LORENTZ]

        self.logit_scale = nn.Parameter(torch.tensor(config.temp))
        self.weight_i2t = self.config.weight_i2t 
        self.curv = torch.as_tensor(config.curv if manifold != EUCLID else 0)
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())
        
    
        if manifold == EUCLID:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.clip_r = None
            self.manifold = Euclidean()
        elif manifold == POINCARE:
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = PoincareBall(c=self.curv, learnable=config.curv_learnable)
        else: 
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable, atol=config.atol, rtol=config.rtol)
        self.manifold_name =  manifold    
        self.model = None 

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

        
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
    
    def itm_loss(self, input_ids, attention_mask, vit_embeds, sim_t2i, sim_i2t):
        text_input_ids_world = input_ids
        bs = input_ids.size(0)
        text_attention_mask_world = attention_mask
        image_embeds_world = vit_embeds
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            mask = (torch.eye(bs) > 0).to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [input_ids, input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [attention_mask, attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            vit_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [vit_embeds, image_embeds_neg, vit_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            vit_embeds.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(vit_embeds.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        return loss_itm
        
    def itc_loss(self, image_embeds , text_embeds):

        sim_q2t = torch.matmul(
            image_embeds.unsqueeze(1), text_embeds.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size, num_query_tokens]
        sim_t2q = torch.matmul(
            text_embeds.unsqueeze(1).unsqueeze(1), image_embeds.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        bs = image_embeds.size(0)
        targets = torch.arange(bs).to(self.device)

        itc_loss= (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": itc_loss.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc": (sim_i2t.argmax(-1) == targets).float().mean().item(),
            "logits/curvature": self.manifold.k.item(),
        }
        return itc_loss, stats, sim_i2t
    
    def postprocess_embeds(self, embed):
        if self.mapper is not None:
            self.manifold.assert_check_point_on_manifold(embed)
            return embed 
        else:
            return F.normalize(embed, p=2, dim=-1) 

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vit_outputs, vision_outputs = self.model(
            pixel_values=pixel_values,
        )

        _, text_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs
        text_embeds = text_outputs
        self.postprocess_embeds(image_embeds)
        self.postprocess_embeds(text_embeds)
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        itm_loss = self.itm_loss(input_ids=input_ids, attention_mask=attention_mask, vit_embeds=vit_outputs, sims_i2t=sims_i2t, sim_t2i=sims_i2t.T)
        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itm_loss + itc_loss 
        return loss, stats

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        _, text_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = self.postprocess_embeds(text_outputs)
        return text_embeds

    def get_vision_features(self, pixel_values:torch.Tensor):
        vit_outputs, vision_outputs = self.model(
            pixel_values=pixel_values,
        )
        vision_embeds = self.postprocess_embeds(vision_outputs)
        return vit_outputs ,vision_embeds


