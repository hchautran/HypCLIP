import torch
import torch.nn as nn
from .modules.text_model import CLIPText, CLIPGraphText
from .modules.vision_model import CLIPVision, CLIPGraphVision
from loralib.share_lora_clip import CLIPTextModelWithProjection as LoraCLIPText
from loralib.share_lora_clip import CLIPVisionModelWithProjection as LoraCLIPVision
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from loralib.utils import mark_only_lora_as_trainable 

from transformers import CLIPConfig 
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel
from peft import get_peft_model, LoraConfig, TaskType

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypCLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypCLIP, self).__init__(config)
        clip_config = CLIPConfig.from_pretrained(self.model_ckt) 
        clip_config.text_config.r =  32 
        clip_config.vision_config.r = 32 

        text_model = LoraCLIPText.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir, config=clip_config.text_config
        )
        vision_model = LoraCLIPVision.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir, config=clip_config.vision_config
        )
        mark_only_lora_as_trainable(model=text_model)
        mark_only_lora_as_trainable(model=vision_model)
        


        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = nn.ModuleList([])
        vision_head = nn.ModuleList([])
        text_head = nn.ModuleList([text_model.text_projection])
        vision_head = nn.ModuleList([vision_model.visual_projection])

        if self.manifold_name !=  EUCLID:
            text_head.append(
                ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            )
            vision_head.append(
                ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            )


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

class HypGraphCLIP(BaseModel):
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
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=[
                'k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2', 'text_projection'
            ]
        )
        vision_target_modules = ['visual_projection']
        for i in range(config.vision_trainable_blocks): 
            index = 11 - i
            vision_target_modules.extend([
                f'*{index}.self_attn.out_proj',
                f'*{index}.self_attn.q_proj',
                f'*{index}.self_attn.k_proj',
                f'*{index}.self_attn.v_proj', 
                f'*{index}.mlp.fc1', 
                f'*{index}.mlp.fc2', 
            ])
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=vision_target_modules
        )
        text_model = get_peft_model(text_model, text_peft_config)
        vision_model = get_peft_model(vision_model, vision_peft_config)
        print(text_model.print_trainable_parameters())
        print(vision_model.print_trainable_parameters())

        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = text_model.text_projection
        vision_head = vision_model.visual_projection
        mapper = None
        if self.manifold_name !=  EUCLID:
            mapper =  ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)


        self.vision_model = CLIPGraphVision(
            config=config,
            body=vision_body,
            head=vision_head,
            mapper=mapper,
            num_layers=config.vision_trainable_blocks,
        )
        self.text_model = CLIPGraphText(
            config=config,
            body=text_body,
            head=text_head,
            manifold_mapper=mapper,
            num_layers=config.text_trainable_blocks,
        )
