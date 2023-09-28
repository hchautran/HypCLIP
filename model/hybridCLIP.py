import torch
import torch.nn as nn
from .modules.text_model import CLIPText
from .modules.vision_model import CLIPVision
from .modules.discriminator import Discriminator as DisModel
from .modules.seq_linear import LorentzSeqLinear, HypSeqLinear
from .modules.hyp_discriminator import HypDiscriminator as HypDisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
# from hyptorch.lorentz.modeling_clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from transformers.models.clip.modeling_clip import CLIPOutput
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block
from .modules.utils import ManifoldMapper, LorentzCentroidPooler
from model.baseHybridModel import BaseModel
from peft import get_peft_model, LoraConfig, TaskType

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypCLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypCLIP, self).__init__(config)

        text_model = CLIPTextModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        text_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1, 
            target_modules=[
                # 'token_embedding',
                'k_proj', 
                'v_proj', 
                'q_proj', 
                'out_proj', 
                'fc1', 
                'fc2', 
                'text_projection'
                ]
        )
        vision_target_modules = ['visual_projection']
        for i in range(config.vision_trainable_blocks): 
            index = 11 - i
            vision_target_modules.extend([
                # 'patch_embedding',
                f'*{index}.mlp.fc1', 
                f'*{index}.mlp.fc2', 
                f'*{index}.self_attn.out_proj',
                f'*{index}.self_attn.q_proj',
                f'*{index}.self_attn.k_proj',
                f'*{index}.self_attn.v_proj', 
            ])
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=8, 
            lora_dropout=0.1, 
            target_modules=vision_target_modules
        )
        text_model = get_peft_model(text_model, text_peft_config)
        print(text_model.print_trainable_parameters())
        vision_model = get_peft_model(vision_model, vision_peft_config)
        print(vision_model.print_trainable_parameters())

        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = nn.ModuleList([])
        vision_head = nn.ModuleList([])
        text_head = nn.ModuleList([text_model.text_projection])
        vision_head = nn.ModuleList([vision_model.visual_projection])



        self.vision_model = CLIPVision(
            config=config,
            body=vision_body,
            head=vision_head,
            num_trainable_blocks=config.vision_trainable_blocks,
            freeze_embedding=config.freeze_embedding,
        )
        self.text_model = CLIPText(
            config=config,
            body=text_body,
            head=text_head,
            num_trainable_blocks=config.text_trainable_blocks,
            freeze_embeddings=config.freeze_embedding,
        )
