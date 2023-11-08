import torch
import torch.nn as nn
from .modules.model import CLIPEncoder 
from .modules.graphs import GraphModel, LorentzGraphModel
from loralib.share_lora_clip import CLIPTextModelWithProjection as LoraCLIPText
from loralib.share_lora_clip import CLIPVisionModelWithProjection as LoraCLIPVision
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from loralib.utils import mark_only_lora_as_trainable 
import torch.nn.functional as F
from transformers import CLIPConfig 
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel
from model.baseQueueModel import BaseModelWithQueue 
from model.baseDistilledModel import BaseModel as DistiledBaseModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
from lavis.models import BlipRetrieval
from .modules.lavis_model import LavisEncoder, LavisBLIPGraphHead, LavisLorentzBLIPGraphHead 

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


def get_lora_clip(config, vision_model, text_model):
    vision_target_modules = ['visual_projection']
    text_target_modules = ['text_projection']
  
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
    for i in range(config.text_trainable_blocks): 
        index = 11 - i
        text_target_modules.extend([
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
        r=64, 
        lora_alpha=64, 
        lora_dropout=0.2, 
        target_modules=vision_target_modules
    )
    text_peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=64, 
        lora_alpha=64, 
        lora_dropout=0.2, 
        target_modules=text_target_modules
    )
    text_lora_model = get_peft_model(text_model, text_peft_config)
    vision_lora_model = get_peft_model(vision_model, vision_peft_config)
    print('trainable params in vision model:',vision_lora_model.print_trainable_parameters())
    print('trainable params in text model:',text_lora_model.print_trainable_parameters())
    return vision_lora_model, text_lora_model




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
        text_head = text_model.text_projection
        vision_head = vision_model.visual_projection
        mapper = None

        if self.manifold_name !=  EUCLID:
            mapper =  ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r) 


        self.vision_model = CLIPEncoder(
            config=config,
            body=vision_body,
            head=vision_head,
            manifold_mapper=mapper
        )
        self.text_model = CLIPEncoder(
            config=config,
            body=text_body,
            head=text_head,
            manifold_mapper=mapper
        )

class HypGraphCLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypGraphCLIP, self).__init__(config)

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
                # 'k_proj', 'v_proj', 'q_proj', 'text_projection'
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
        text_body = get_peft_model(text_model, text_peft_config)
        vision_body = get_peft_model(vision_model, vision_peft_config)
        print(text_body.print_trainable_parameters())
        print(vision_body.print_trainable_parameters())

        text_body = text_model.text_model
        vision_body = vision_model.vision_model

        text_head = text_model.text_projection
        vision_head = vision_model.visual_projection
        mapper = None
        if self.manifold_name !=  EUCLID:
            mapper =  ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
            self.vision_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                graph_hidden_channels=config.graph_hidden_channels,
                manifold_mapper=mapper,
                num_layers=config.num_vision_hidden_states,
                hidden_size=512,
                num_hidden_layers=6,
            )
            self.text_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=mapper,
                graph_hidden_channels=config.graph_hidden_channels,
                num_layers=config.num_text_hidden_states,
                hidden_size=512,
                num_hidden_layers=6
            )
        else:
            self.vision_model = GraphModel(
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=mapper,
                num_layers=config.num_vision_hidden_states,
                graph_hidden_channels=config.graph_hidden_channels,
                hidden_size=512,
                num_hidden_layers=6
            )
            self.text_model = GraphModel(
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=mapper,
                num_layers=config.num_text_hidden_states,
                graph_hidden_channels=config.graph_hidden_channels,
                hidden_size=512,
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
    ) -> Union[Tuple, CLIPOutput]:

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
            # if self.config.use_margin_loss:
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


class HypCLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypCLIPWithQueue, self).__init__(config)
        clip_config = CLIPConfig.from_pretrained(self.model_ckt) 
        clip_config.text_config.r =  32 
        clip_config.vision_config.r = 32 

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
        mapper = None

        if self.config.manifold !=  EUCLID:
            mapper =ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.vision_model = CLIPEncoder(
            config=config,
            body=vision_body,
            head=vision_head,
            manifold_mapper=mapper
        )
        self.text_model = CLIPEncoder(
            config=config,
            body=text_body,
            head=text_head,
            manifold_mapper=mapper
        )        

        self._init_queue(config, vision_model.config.projection_dim)
        
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
            self.vision_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_vision_hidden_states,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                shared_proj_layers=config.shared_proj_layers,
                use_root=config.use_root
            )
            self.text_model = LorentzGraphModel(
                manifold=self.manifold,
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_text_hidden_states,
                graph_hidden_channels=config.graph_hidden_channels,
                hidden_size=config.proj_layer_hidden_sizes,
                num_hidden_layers=config.num_proj_layers,
                shared_proj_layers=config.shared_proj_layers,
                use_root=config.use_root
            )
        else:
            self.vision_model = GraphModel(
                ft_in=vision_model.config.hidden_size,
                ft_out=vision_model.config.projection_dim,
                config=config,
                body=vision_body,
                head=vision_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_vision_hidden_states,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                shared_proj_layers=config.shared_proj_layers,
                use_root=config.use_root
            )
            self.text_model = GraphModel(
                ft_in=text_model.config.hidden_size,
                ft_out=text_model.config.projection_dim,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=config.num_text_hidden_states,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
                shared_proj_layers=config.shared_proj_layers,
                use_root=config.use_root
            )
        
        self._init_queue(config, vision_model.config.projection_dim)






class HypCLIPDistilled(DistiledBaseModel):
    def __init__(self, config, teacher_model:BlipRetrieval) -> None:
        super(HypCLIPDistilled, self).__init__(config)
        clip_config = CLIPConfig.from_pretrained(self.model_ckt) 
        clip_config.text_config.r =  32 
        clip_config.vision_config.r = 32 

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
        mapper = None

        if self.config.manifold !=  EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)


        self.vision_model = CLIPEncoder(
            config=config,
            body=vision_body,
            head=vision_head,
            mapper=mapper
        )
        self.text_model = CLIPEncoder(
            config=config,
            body=text_body,
            head=text_head,
            mapper=mapper
        )        
        self.vision_teacher = LavisEncoder(
            config,
            body=teacher_model.visual_encoder,
            head=teacher_model.vision_proj,
            mapper=None,
            use_normalized=config.normalize_image_embed
        )
        self.text_teacher = LavisEncoder(
            config,
            body= teacher_model.text_encoder,
            head=teacher_model.text_proj,
            mapper=None,
            use_normalized=config.normalize_text_embed
        )
