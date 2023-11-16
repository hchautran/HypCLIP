import torch
import torch.nn as nn
from .modules.model import BLIPEncoder
from .modules.blip import LavisEncoder, LavisBLIPGraphHead, LavisLorentzBLIPGraphHead 
from transformers import BlipForImageTextRetrieval
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel 
from model.baseQueueModel import BaseModelWithQueue 
from model.baseDistilledModel import BaseModel as BaseDistilModel 
from .modules.utils import ManifoldMapper
import torch.nn.functional as F
from lavis import Blip2Qformer
from copy import deepcopy
from .modules.graphs import GraphModel, LorentzGraphModel
EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"

def get_lora_lavis_blip(config, model):
    return



class LavisHypGraphBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model) -> None:
        super(LavisHypGraphBLIPWithQueue, self).__init__(config)
        
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        text_body = model.text_encoder
        vision_body = model.visual_encoder
        text_head = model.text_proj
        vision_head = model.vision_proj
        mapper = None
        if config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
            self.vision_model = LavisLorentzBLIPGraphHead(
                manifold=self.manifold,
                ft_in=768,
                ft_out=256,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )
            self.text_model = LavisLorentzBLIPGraphHead(
                manifold=self.manifold,
                ft_in=768,
                ft_out=256,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )
        else:
            self.vision_model = LavisBLIPGraphHead(
                ft_in=768,
                ft_out=256,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )
            self.text_model = LavisBLIPGraphHead(
                ft_in=768,
                ft_out=256,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )

        # self.eu_logit_scale = model.temp
        # self.logit_scale = model.temp
        self._init_queue(config, 256)

