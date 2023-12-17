import torch
import torch.nn as nn
from transformers import AutoModel 
from .modules.dct_blip import BLIPEncoder, DCTLAVISBlip, DCTHFClip, FreqPredictor 
from peft import get_peft_model, LoraConfig, TaskType
from typing import  Optional, Tuple, Union
from .modules.utils import freeze_blip
from model.baseQueueModel import BaseModelWithQueue 
from lavis.models import load_model_and_preprocess
import math

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
def get_lora_clip(config, model):
    target_modules = [
        'visual_projection',
        'text_projection'
    ]
    max_len = 11 if 'base' in config.model_ckt else 23
  
    for i in range(config.text_trainable_blocks): 
        index = max_len - i
        target_modules.extend([
            f'*{index}.self_attn.out_proj',
            f'*{index}.self_attn.q_proj',
            f'*{index}.self_attn.k_proj',
            f'*{index}.self_attn.v_proj', 
            f'*{index}.mlp.fc1', 
            f'*{index}.mlp.fc2', 
        ])
    for i in range(config.vision_trainable_blocks): 
        index = max_len - i
        target_modules.extend([
            f'*{index}.self_attn.out_proj',
            f'*{index}.self_attn.q_proj',
            f'*{index}.self_attn.k_proj',
            f'*{index}.self_attn.v_proj', 
            f'*{index}.mlp.fc1', 
            f'*{index}.mlp.fc2', 
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
    print('trainable params model:',model.print_trainable_parameters())
    return model 




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
        lora_dropout=0.3, 
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config) 
    model.print_trainable_parameters()
    return model

class BLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(BLIPWithQueue, self).__init__(config)

        model = AutoModel.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
        # model = get_lora_blip(config, model=model) 
        self.model = BLIPEncoder(
           text_body=model.text_model, 
           vision_body=model.vision_model, 
           text_head=model.text_projection,
           vision_head=model.visual_projection, 
        )
        
        self._init_queue(config, model.config.image_text_hidden_size)


class DCTHFWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(DCTHFWithQueue, self).__init__(config)
        model = AutoModel.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
        if 'clip' in config.model_ckt:
            model = get_lora_clip(config, model=model) 
            self.model = DCTHFClip(model)
        else:
            model = get_lora_blip(config, model=model) 
            self.model = DCTHFClip(model)

        
        self._init_queue(config, model.config.projection_dim)
    
    def get_vision_features(self, pixel_values: torch.Tensor, apply_fourier=True):
        image_output = self.model.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)
        image_feat = self.postprocess_embeds(image_output[1])
        return image_feat, image_output[0]
    
class DCTLAVISLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model) -> None:
        super(DCTLAVISLIPWithQueue, self).__init__(config)
        model = get_lora_lavis_blip(config, model=model) 
        self.model = DCTLAVISBlip(model)
        ori_len = 576
        final_len = 576
        for r in self.model.r_list:
            final_len = math.ceil(final_len * r)
        self.freq_predictor = FreqPredictor(final_len, 512, ori_len)
        
        self._init_queue(config, 256)
    
    def get_vision_features(self, pixel_values: torch.Tensor, apply_fourier=True):
        image_output = self.model.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)
        image_feat = self.postprocess_embeds(image_output[1])
        return image_feat, image_output[0]
