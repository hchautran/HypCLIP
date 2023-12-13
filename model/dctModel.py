import torch
import torch.nn as nn
from transformers import BlipForImageTextRetrieval 
from .modules.dct_blip import DCTBlipForImageTextRetrieval, BLIPEncoder 
from peft import get_peft_model, LoraConfig, TaskType
from typing import  Optional, Tuple, Union
from .modules.utils import freeze_blip
from model.baseQueueModel import BaseModelWithQueue 

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


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


class BLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(BLIPWithQueue, self).__init__(config)

        model = BlipForImageTextRetrieval.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
        # model = get_lora_blip(config, model=model) 
        self.model = BLIPEncoder(
           text_body=model.text_encoder, 
           vision_body=model.vision_model, 
           text_head=model.text_proj,
           vision_head=model.vision_proj, 
        )

        
        self._init_queue(config, model.config.image_text_hidden_size)


class DCTBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(DCTBLIPWithQueue, self).__init__(config)
        model = BlipForImageTextRetrieval.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
        model = get_lora_blip(config, model=model) 
        self.model = DCTBlipForImageTextRetrieval(model)

        
        self._init_queue(config, 256)
    
    def get_vision_features(self, pixel_values: torch.Tensor, apply_fourier=True):
        image_output = self.model.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)
        image_feat = self.postprocess_embeds(image_output[1])
        return image_feat, image_output[0]
    


