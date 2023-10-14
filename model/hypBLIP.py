import torch
import torch.nn as nn
from .modules.text_model import BLIPText, BLIPGraphText
from .modules.vision_model import BLIPVision, BLIPGraphVision
from .modules.lavis_model import LavisEncoder, BLIPGraphHead, LorentzBLIPGraphHead 
from transformers import BlipForImageTextRetrieval, CLIPTextModelWithProjection, CLIPVisionModelWithProjection 
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel 
from model.baseQueueModel import BaseModelWithQueue 
from .modules.utils import ManifoldMapper
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from lavis.models import BlipRetrieval
from copy import deepcopy

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"

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
        self.config = config
        model = get_peft_model(model, peft_config) 
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = nn.ModuleList([model.text_proj])
        vision_head = nn.ModuleList([model.vision_proj])
   
        text_head.append(
            ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
        )
        vision_head.append(
            ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
        )


        self.vision_model = BLIPVision(
            config,
            body=vision_body,
            head=vision_head,
            # num_trainable_blocks=config.vision_trainable_blocks,
            # freeze_embedding=config.freeze_embedding,
        )
        self.text_model = BLIPText(
            config,
            body=text_body,
            head=text_head,
            # num_trainable_blocks=config.text_trainable_blocks,
            # freeze_embeddings=config.freeze_embedding,
        )

class LavisBLIP(BaseModel):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIP, self).__init__(config)
        
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
                '*.11.self_attn.qkv',
                '*.11.self_attn.proj',
                '*.11.mlp.fc1',
                '*.11.mlp.fc2',
                '*.10.self_attn.qkv',
                '*.10.self_attn.proj',
                '*.10.mlp.fc1',
                '*.10.mlp.fc2',
            ]
        )
        self.config = config
        model = get_peft_model(model, peft_config) 
        text_body = model.text_encoder
        vision_body = model.visual_encoder
        text_head = nn.ModuleList([model.text_proj])
        vision_head = nn.ModuleList([model.vision_proj])
        if self.manifold_name != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            text_head.append(mapper)
            vision_head.append(mapper)


        self.vision_model = LavisEncoder(
            config,
            body=vision_body,
            head=vision_head,
        )
        self.text_model = LavisEncoder(
            config,
            body=text_body,
            head=text_head,
        )

    def forward(self, input_ids: torch.LongTensor | None = None, pixel_values: torch.FloatTensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None): 
        return super().forward(input_ids, pixel_values, attention_mask, position_ids)
    
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None):
        return super().get_text_features(input_ids, attention_mask, position_ids)
    
    def get_vision_features(self, pixel_values: torch.Tensor):
        return super().get_vision_features(pixel_values)


class HypGraphBLIP(BaseModel):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(HypGraphBLIP, self).__init__(config)
        
        self.config = config
        text_body = model.text_encoder
        vision_body = model.visual_encoder
        text_head = model.text_proj
        vision_head = model.vision_proj
        mapper = None
        if self.manifold_name != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            self.vision_model = LorentzBLIPGraphHead(
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
            self.text_model = LorentzBLIPGraphHead(
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
            self.vision_model = BLIPGraphHead(
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
            self.text_model = BLIPGraphHead(
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


    def eval(self):
        self.vision_model.eval()
        self.text_model.eval()

    def train(self):
        self.vision_model.train()
        self.text_model.train()

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

        
class HypGraphCLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypGraphCLIPWithQueue, self).__init__(config)

        text_model = CLIPTextModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        vision_model, text_model = get_lora_clip(config, vision_model=vision_model, text_model=text_model)
      
        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = text_model.text_projection
        vision_head = vision_model.visual_projection

        if self.config.manifold !=  EUCLID:
            self.vision_model = LorentzCLIPGraphHead(
                manifold=self.manifold,
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_vision_hidden_states,
                hidden_size=512,
                num_hidden_layers=6,
            )
            self.text_model = LorentzCLIPGraphHead(
                manifold=self.manifold,
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_text_hidden_states,
                hidden_size=512,
                num_hidden_layers=6
            )
        else:
            self.vision_model = CLIPGraphHead(
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_vision_hidden_states,
                hidden_size=512,
                num_hidden_layers=6
            )
            self.text_model = CLIPGraphHead(
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_text_hidden_states,
                hidden_size=512,
                num_hidden_layers=6,
            )
        self.vision_model_m= deepcopy(self.vision_model) 
        self.text_model_m= deepcopy(self.text_model) 
        self.model_pairs = [
            [self.vision_model, self.vision_model_m],
            [self.text_model, self.text_model_m],
        ]
        self.copy_params()
        self.ft_out = vision_model.config.projection_dim if config.manifold != LORENTZ else vision_model.config.projection_dim + 1 
          # create the queue
        if config.manifold == EUCLID:
            self.register_buffer("image_queue", torch.randn(self.queue_size, self.ft_out).T)
            self.register_buffer("text_queue", torch.randn(self.queue_size, self.ft_out).T)
            self.image_queue = nn.functional.normalize(self.image_queue, dim=1)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=1)
        else:
            self.register_buffer("image_queue", self.manifold.random(self.queue_size, self.ft_out).T)
            self.register_buffer("text_queue", self.manifold.random(self.queue_size, self.ft_out).T)
            self.manifold.assert_check_point_on_manifold(self.image_queue.T)
            self.manifold.assert_check_point_on_manifold(self.text_queue.T)

        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def eval(self):
        self.vision_model_m.eval()
        self.text_model_m.eval()
        self.vision_model_m.eval()
        self.text_model_m.eval()

    def train(self):
        self.vision_model.train()
        self.text_model.train()
        self.vision_model_m.train()
        self.text_model_m.train()