import torch
import torch.nn as nn
from .modules.text_model import BLIPText
from .modules.vision_model import BLIPVision
from .modules.discriminator import Discriminator as DisModel
from .modules.seq_linear import LorentzSeqLinear, HypSeqLinear
from .modules.hyp_discriminator import HypDiscriminator as HypDisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .manifolds.euclidean import Euclidean
from .manifolds.hyperboloid import Hyperboloid
from .manifolds.lorentz import Lorentz
from .manifolds.poincare import PoincareBall
from transformers import BlipForImageTextRetrieval
from typing import Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
from .modules.utils import ManifoldMapper
from .modules.utils import ManifoldMapper, LorentzCentroidPooler
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block
from transformers.activations import ACT2FN
from peft import get_peft_model, LoraConfig, TaskType
from model.baseHybridModel import BaseModel


EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypBLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypBLIP, self).__init__(config)

        model = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1, 
            target_modules=[
                'dense', 
                'query', 
                'value',
                'key', 
                'text_proj', 
                'vision_proj',
                '*.11.self_attn.qkv'
                '*.11.self_attn.projection'
                '*.11.mlp.fc1'
                '*.11.mlp.fc2'
            ]
        )
        print(model)
        model = get_peft_model(model, peft_config) 
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = nn.ModuleList([model.text_proj])
        vision_head = nn.ModuleList([model.vision_proj])

        self.vision_model = BLIPVision(
            config,
            body=vision_body,
            head=vision_head,
        )
        self.text_model = BLIPText(
            config,
            body=text_body,
            head=text_head,
        )
