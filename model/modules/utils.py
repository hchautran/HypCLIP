from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import BlipVisionModel, BlipTextModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt.manifolds.stereographic import PoincareBall 
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoProcessor, BlipModel, CLIPModel
from .fuseModel import LavisEncoder, CLIPEncoder, BLIPEncoder
from lavis.models import load_model_and_preprocess
from typing import List
from typing import Union

def fr(m):
    for param in m.parameters():
        param.requires_grad = False


def freeze_clip(
    vision_model: CLIPVisionModelWithProjection = None,
    text_model: CLIPTextModelWithProjection = None,
    freeze_embeddings=True,
    num_trainable_blocks=-1,
):
    if num_trainable_blocks == -1:
        return

    if vision_model is not None:
        if freeze_embeddings:
            fr(vision_model.encoder.embeddings)
        for idx in range(len(vision_model.encoder.layers) - num_trainable_blocks):
            fr(vision_model.encoder.layers[idx])

    if text_model is not None:
        if freeze_embeddings:
            fr(text_model.encoder.embeddings)
        for idx in range(len(text_model.encoder.layers) - num_trainable_blocks):
            fr(text_model.encoder.layers[idx])


def freeze_blip(
    vision_model: BlipVisionModel = None,
    text_model: BlipTextModel = None,
    vision_head: BlipVisionModel = None,
    text_head: BlipTextModel = None,
    freeze_embeddings=True,
    num_trainable_blocks=0,
):
    if num_trainable_blocks == -1:
        return

    if vision_model is not None:
        if freeze_embeddings:
            fr(vision_model.embeddings)
        if vision_head is not None:
            fr(vision_head)
        for idx in range(len(vision_model.encoder.layers) - num_trainable_blocks):
            fr(vision_model.encoder.layers[idx])
            

    if text_model is not None:
        if freeze_embeddings:
            fr(text_model.embeddings)
        if text_head is not None:
            fr(text_head)
        for idx in range(len(text_model.encoder.layer) - num_trainable_blocks):
            fr(text_model.encoder.layer[idx])


class ManifoldMapper(nn.Module):
    def __init__(self, manifold:Union[PoincareBall,CustomLorentz], curv, clip_r=None):
        super().__init__()
        self.manifold = manifold
        self.clip_r = clip_r

    def forward(self, x, use_normalized=False):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
            x = x * fac
        
        if isinstance(self.manifold, CustomLorentz): 
            if use_normalized:
                x = F.normalize(x, p=2, dim=-1) 
            x = F.pad(x, (1,0), "constant", 0)
            out = self.manifold.expmap0(x)
        else:
            out = self.manifold.expmap0(x)
        return out 


class LorentzCentroidPooler(nn.Module):
    def __init__(self, manifold: CustomLorentz, curv, clip_r=None):
        super().__init__()
        self.manifold = manifold
        self.curv = curv
        self.clip_r = clip_r

    def forward(self, x, w=None ):
        pooled_x = self.manifold.centroid(x, w)
        return pooled_x


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

def get_lora_blip(config, model):

    target_modules = [ 
        'text_projection', 
        'visual_projection',
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

def get_lora_clip(config, model):
    target_modules = ['visual_projection', 'text_projection']
  
    for i in range(config.vision_trainable_blocks): 
        index = 11 - i
        target_modules.extend([
            f'*{index}.self_attn.out_proj',
            f'*{index}.self_attn.q_proj',
            f'*{index}.self_attn.k_proj',
            f'*{index}.self_attn.v_proj', 
            f'*{index}.mlp.fc1', 
            f'*{index}.mlp.fc2', 
        ])
    for i in range(config.text_trainable_blocks): 
        index = 11 - i
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
    lora_model = get_peft_model(model, peft_config)
    return lora_model 




def prepare_processors_and_models(model_ckts:List[str]):
    tokenizers = []
    vis_processors = []
    txt_processors = []
    models = []
    
    for i, model_ckt in enumerate(model_ckts):
        if 'lavis' in model_ckt:
            model, vis_processor, txt_processor = load_model_and_preprocess(
                "blip_retrieval", 
                "coco" if 'coco' in model_ckt else 'flickr', 
                is_eval=True
            )
            tokenizers.append(model.tokenizer)
            vis_processors.append(vis_processor['eval'])
            txt_processors.append(txt_processor['eval'])
            models.append(model)
        else:
            tokenizers.append(AutoProcessor.from_pretrained(model_ckt))
            vis_processors.append(AutoProcessor.from_pretrained(model_ckt))
            txt_processors.append(None)
            models.append(AutoModel.from_pretrained(model_ckt))
    return tokenizers, vis_processors, txt_processors, models

def prepare_encoder(config, models):
    vis_encoders = [] 
    text_encoders = [] 
    d_visions = []
    d_texts= []
    text_head = None
    vision_head = None
    for i, model in enumerate(models):
        if isinstance(model, CLIPModel): 
            model = get_lora_clip(config, model)
            if i == 0:
                text_head = model.text_projection
                vision_head = model.visual_projection
                
            vis_encoders.append(CLIPEncoder(model.vision_model, model.visual_projection)) 
            text_encoders.append(CLIPEncoder(model.text_model, model.text_projection)) 
            d_visions.append(model.config.vision_config.hidden_size)
            d_texts.append(model.config.text_config.hidden_size)
        elif isinstance(model, BlipModel): 
            model = get_lora_blip(config, model)
            if i == 0:
                text_head = model.text_projection
                vision_head = model.visual_projection
            vis_encoders.append(BLIPEncoder(model.vision_model, model.visual_projection)) 
            text_encoders.append(BLIPEncoder(model.text_model, model.text_projection)) 
            d_visions.append(model.config.vision_config.hidden_size)
            d_texts.append(model.config.text_config.hidden_size)
        else:
            model = get_lora_lavis_blip(config, model)
            if i == 0:
                text_head = model.text_proj
                vision_head = model.vision_proj
            vis_encoders.append(LavisEncoder(model.visual_encoder, model.vision_proj)) 
            text_encoders.append(LavisEncoder(model.text_encoder, model.text_proj)) 
            d_visions.append(768)
            d_texts.append(768)
    return vis_encoders, text_encoders, text_head, vision_head, d_visions, d_texts 