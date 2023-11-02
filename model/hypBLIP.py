import torch
import torch.nn as nn
from .modules.model import BLIPEncoder
from .modules.lavis_model import LavisEncoder, LavisBLIPGraphHead, LavisLorentzBLIPGraphHead 
from transformers import BlipForImageTextRetrieval
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel 
from model.baseQueueModel import BaseModelWithQueue 
from model.baseDistilledModel import BaseModel as BaseDistilModel 
from .modules.utils import ManifoldMapper
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from lavis.models import BlipRetrieval
from copy import deepcopy
from .modules.graphs import GraphModel, LorentzGraphModel
EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"

def get_lora_blip(config, model):

    target_modules = [ 
        'text_proj', 
        'vision_proj',
    ]
    for i in range(config.vision_trainable_blocks): 
        index = 11 - i
        target_modules.extend([
            f'*{index}.self_attn.qkv',
            f'*{index}.self_attn.projection',
            f'*{index}.mlp.fc1', 
            f'*{index}.mlp.fc2', 
        ])
    for i in range(config.text_trainable_blocks): 
        index = 11 - i
        target_modules.extend([
            f'*{index}.attention.output.dense', 
            f'*{index}.attention.self.query', 
            f'*{index}.attention.self.value',
            f'*{index}.attention.self.key', 
        ])
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.2, 
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config) 
    model.print_trainable_parameters()
    return model


def get_lora_lavis_blip(config, model):

    target_modules = [ 
        'text_proj', 
        'vision_proj',
    ]
    for i in range(config.vision_trainable_blocks): 
        index = 11 - i
        target_modules.extend([
            f'*{index}.attn.qkv',
            f'*{index}.attn.proj',
            f'*{index}.mlp.fc1', 
            f'*{index}.mlp.fc2', 
        ])
    for i in range(config.text_trainable_blocks): 
        index = 11 - i
        target_modules.extend([
            f'*{index}.attention.output.dense', 
            f'*{index}.attention.self.query', 
            f'*{index}.attention.self.value',
            f'*{index}.attention.self.key', 
        ])
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.2, 
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config) 
    model.print_trainable_parameters()
    return model



class HypBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypBLIPWithQueue, self).__init__(config)
        self.config = config

        model = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        model = get_lora_blip(config, model)
      
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = model.text_proj
        vision_head = model.vision_proj
        mapper = None
   
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.vision_model = BLIPEncoder(
            config,
            body=vision_body,
            head=vision_head,
            manifold_mapper=mapper
        )
        self.text_model = BLIPEncoder(
            config,
            body=text_body,
            head=text_head,
            manifold_mapper=mapper
        )
        self._init_queue(config, 256)

class LavisBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIPWithQueue, self).__init__(config)
        
    
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.vision_model = LavisEncoder(
            config,
            body=model.visual_encoder,
            head=model.vision_proj,
            mapper=mapper,
            use_normalized=config.normalize_image_embed
        )
        self.text_model = LavisEncoder(
            config,
            body=model.text_encoder,
            head=model.text_proj,
            mapper=mapper,
            use_normalized=config.normalize_text_embed
        )
        self._init_queue(config, 256)
    

class LavisBLIP(BaseModel):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIP, self).__init__(config)
        
    
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.vision_model = LavisEncoder(
            config,
            body=model.visual_encoder,
            head=model.vision_proj,
            mapper=mapper,
            use_normalized=config.normalize_image_embed
        )
        self.text_model = LavisEncoder(
            config,
            body=model.text_encoder,
            head=model.text_proj,
            mapper=mapper,
            use_normalized=config.normalize_text_embed
        )
    

 
 
class DistilLavisBLIP(BaseDistilModel):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(DistilLavisBLIP, self).__init__(config)
        teacher_model = deepcopy(model)

        for param in teacher_model.parameters():
            param.requires_grad = False
    
        model = get_lora_blip(config=config, model=model) 
        self.config = config
        mapper = None
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
        if not self.config.use_graph :
            self.vision_model = LavisEncoder(
                config,
                body=model.visual_encoder,
                head=model.vision_proj,
                mapper=mapper,
                use_normalized=config.normalize_image_embed
            )
            self.text_model = LavisEncoder(
                config,
                body=model.text_encoder,
                head=model.text_proj,
                mapper=mapper,
                use_normalized=config.normalize_text_embed
            )
        else:
            if config.manifold != EUCLID:
                mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
                self.vision_model = LavisLorentzBLIPGraphHead(
                    manifold=self.manifold,
                    ft_in=768,
                    ft_out=256,
                    config=config,
                    body=model.visual_encoder,
                    head=model.vision_proj,
                    manifold_mapper=mapper,
                    num_layers=1,
                    hidden_size=config.proj_layer_hidden_sizes,
                    num_hidden_layers=config.num_proj_layers,
                )
                self.text_model = LavisLorentzBLIPGraphHead(
                    manifold=self.manifold,
                    ft_in=768,
                    ft_out=256,
                    config=config,
                    body=model.text_encoder,
                    head=model.text_proj,
                    manifold_mapper=mapper,
                    num_layers=1,
                    hidden_size=config.proj_layer_hidden_sizes,
                    num_hidden_layers=config.num_proj_layers,
                )
            else:
                self.vision_model = LavisBLIPGraphHead(
                    ft_in=768,
                    ft_out=256,
                    config=config,
                    body=model.visual_encoder,
                    head=model.vision_proj,
                    manifold_mapper=mapper,
                    num_layers=1,
                    hidden_size=config.proj_layer_hidden_sizes,
                    num_hidden_layers=config.num_proj_layers,
                )
                self.text_model = LavisBLIPGraphHead(
                    ft_in=768,
                    ft_out=256,
                    config=config,
                    body=model.text_encoder,
                    head=model.text_proj,
                    manifold_mapper=mapper,
                    num_layers=1,
                    hidden_size=config.proj_layer_hidden_sizes,
                    num_hidden_layers=config.num_proj_layers,
                )


        
        self.vision_teacher = LavisEncoder(
            config,
            body=teacher_model.visual_encoder,
            head=teacher_model.vision_proj,
            mapper=None,
        )
        self.text_teacher = LavisEncoder(
            config,
            body=teacher_model.text_encoder,
            head=teacher_model.text_proj,
            mapper=None,
        )
 


class LavisHypGraphBLIP(BaseModel):
    def __init__(self, config, model) -> None:
        super(LavisHypGraphBLIP, self).__init__(config)
        
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
            )

        # self.eu_logit_scale = model.temp
        # self.logit_scale = model.temp
        
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
            )

        # self.eu_logit_scale = model.temp
        # self.logit_scale = model.temp
        self._init_queue(config, 256)


class HypGraphBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypGraphBLIPWithQueue, self).__init__(config)

        model = BlipForImageTextRetrieval.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
        model = get_lora_blip(config, model=model) 
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = model.text_proj
        vision_head = model.vision_proj

        if self.config.manifold !=  EUCLID:
            self.vision_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=model.config.vision_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
            self.text_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=model.config.text_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
        else:
            self.vision_model = GraphModel(
                ft_in=model.config.vision_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
            self.text_model = GraphModel(
                ft_in=model.config.text_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
        
        self._init_queue(config, model.config.image_text_hidden_size)




