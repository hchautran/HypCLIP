import torch
import torch.nn as nn
from .modules.text_model import BLIPText, BLIPGraphText
from .modules.vision_model import BLIPVision, BLIPGraphVision
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
        lora_dropout=0.1, 
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config) 
    return model




class HypBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypBLIPWithQueue, self).__init__(config)
        self.config = config

        model = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
      
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = nn.ModuleList([model.text_proj])
        vision_head = nn.ModuleList([model.vision_proj])
   
        text_head.append(
            ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
        )
        vision_head.append(
            ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
        )


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
        self._init_queue(config, 256)

class LavisBLIP(BaseModelWithQueue):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIP, self).__init__(config)
        
    
        model = get_lora_blip(config=config,model=model) 
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
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisHypGraphBLIP, self).__init__(config)
        
        self.config = config
        text_body = model.text_encoder
        vision_body = model.visual_encoder
        text_head = model.text_proj
        vision_head = model.vision_proj
        mapper = None
        if self.manifold_name != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            self.vision_model = LavisLorentzBLIPGraphHead(
                manifold=self.manifold,
                ft_in=768,
                ft_out=256,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=256,
                num_hidden_layers=6,
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
                hidden_size=256,
                num_hidden_layers=6
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
                hidden_size=256,
                num_hidden_layers=6
            )
            self.text_model = LavisBLIPGraphHead(
                ft_in=768,
                ft_out=256,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=256,
                num_hidden_layers=6,
            )



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        graph_image_embeds = vision_outputs[2]
        graph_text_embeds = text_outputs[2]
        self.manifold.assert_check_point_on_manifold(image_embeds)
        self.manifold.assert_check_point_on_manifold(text_embeds)
        self.manifold.assert_check_point_on_manifold(graph_image_embeds)
        self.manifold.assert_check_point_on_manifold(graph_text_embeds)
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds, graph_image_embeds, graph_text_embeds)
        itm_loss = self.itm_loss(image_embeds, text_embeds, sims_i2t=sims_i2t)
        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itm_loss + itc_loss 
        return loss, stats, itc_loss, itm_loss

    def itc_loss(self, image_embeds , text_embeds, graph_image, graph_text):
        bsize = text_embeds.shape[0]
        margin_loss = torch.tensor(0.0)
        g_margin_loss = torch.tensor(0.0)
        g_itc_loss= torch.tensor(0.0)
        eye_mask = torch.eye(bsize).to(self.device) * 1e9
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        target = torch.arange(bsize).to(self.device)
        _scale = self.logit_scale.exp()


        if self.config.use_graph_loss:
            eu_sims_i2t, sims_i2t = self.dist_func(graph_image, graph_text) 
            eu_sims_t2i, sims_t2i = eu_sims_i2t.T, sims_i2t.T
            eu_sims_i2i, sims_i2i = self.dist_func(graph_image, graph_image) 
            eu_sims_t2t, sims_t2t = self.dist_func(graph_text, graph_text) 
            # logits_i2t = torch.cat([sims_i2t * _scale, sims_i2i* _scale - eye_mask], dim=1)
            # logits_t2i = torch.cat([sims_t2i * _scale, sims_t2t* _scale - eye_mask], dim=1)
            g_margin_loss = 0.5 * (self.margin_loss(eu_sims_i2t, eu_sims_t2t - eye_mask) + self.margin_loss(eu_sims_t2i, eu_sims_i2i - eye_mask))
            # g_itc_loss =  self.weight_i2t * F.cross_entropy(logits_i2t, target) + (1 - self.weight_i2t) * F.cross_entropy(logits_t2i, target) 

        eu_sims_i2t, sims_i2t = self.dist_func(image_embeds, text_embeds) 
        eu_sims_t2i, sims_t2i = eu_sims_i2t.T, sims_i2t.T
        eu_sims_i2i, sims_i2i = self.dist_func(image_embeds, image_embeds) 
        eu_sims_t2t, sims_t2t = self.dist_func(text_embeds, text_embeds) 
        logits_i2t = torch.cat([sims_i2t * _scale, sims_t2t* _scale - eye_mask], dim=1)
        logits_t2i = torch.cat([sims_t2i * _scale, sims_i2i* _scale - eye_mask], dim=1)
        if self.config.use_margin_loss:
            margin_loss = 0.5 * (self.margin_loss(eu_sims_i2t, eu_sims_t2t - eye_mask) + self.margin_loss(eu_sims_t2i, eu_sims_i2i - eye_mask))
        itc_loss =  self.weight_i2t * F.cross_entropy(logits_i2t, target) + (1 - self.weight_i2t) * F.cross_entropy(logits_t2i, target) 
        loss = itc_loss + margin_loss + g_itc_loss + g_margin_loss
        
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/margin_loss": margin_loss.item(),
            "logits/itc_loss": itc_loss.item(),
            "logits/g_margin_loss": g_margin_loss.item(),
            "logits/g_itc_loss": g_itc_loss.item(),
            "logits/min": sims_i2t.min().item(),
            "logits/mean": sims_i2t.mean().item(),
            "logits/max": sims_i2t.max().item(),
            "logits/acc": (sims_i2t.argmax(-1) == target).float().mean().item(),
            "logits/eu_acc": (eu_sims_i2t.argmax(-1) == target).float().mean().item(),
            # "logits/curvature": self.manifold.k.item(),
        }
        return loss, stats, sims_i2t

        
class LavisHypGraphBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model) -> None:
        super(LavisHypGraphBLIPWithQueue, self).__init__(config)
        
        model = get_lora_blip(config=config,model=model) 
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
                num_hidden_layers=config.num_proj_layers,
            )

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
                hidden_size=256,
                num_hidden_layers=6,
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
                hidden_size=256,
                num_hidden_layers=6
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
                hidden_size=256,
                num_hidden_layers=6
            )
            self.text_model = GraphModel(
                ft_in=model.config.text_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=256,
                num_hidden_layers=6,
            )
        
        self._init_queue(config, model.config.image_text_hidden_size)




