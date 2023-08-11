import torch
import torch.nn as nn
from .modules.text_model import CLIPText
from .modules.vision_model import CLIPVision 
from .modules.discriminator import Discriminator as DisModel
from .modules.seq_linear import  LorentzSeqLinear, HypSeqLinear
from .modules.hyp_discriminator import HypDiscriminator as HypDisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .manifolds.euclidean import Euclidean 
from .manifolds.hyperboloid import Hyperboloid 
from .manifolds.lorentz import Lorentz 
from .manifolds.poincare import PoincareBall 
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
from .modules.utils import ManifoldMapper 
from model.baseModel import BaseModel



EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'


class HypCLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypCLIP, self).__init__(config)
    

        text_model = CLIPTextModelWithProjection.from_pretrained(self.model_ckt, cache_dir=config.cache_dir) 
        vision_model = CLIPVisionModelWithProjection.from_pretrained(self.model_ckt, cache_dir=config.cache_dir) 
        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = nn.ModuleList([text_model.text_projection])
        vision_head = nn.ModuleList([vision_model.visual_projection])

        if self.manifold_name == LORENTZ: 
            text_head.append(ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r))
            vision_head.append(ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r))
            text_head.append(LorentzSeqLinear(manifold=self.manifold, ft_in=text_body.config.projection_dim, layer_dims=[self.ft_out]))
            vision_head.append(LorentzSeqLinear(manifold=self.manifold, ft_in=vision_body.config.projection_dim, layer_dims=[self.ft_out]))

        self.vision_model = CLIPVision(body=vision_body, head=vision_head, num_trainable_blocks=config.vision_trainable_blocks, freeze_embedding=config.freeze_embedding)
        self.text_model = CLIPText(body=text_body, head=text_head, num_trainable_blocks=config.text_trainable_blocks, freeze_embeddings=config.freeze_embedding)




    