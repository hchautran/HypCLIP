import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval
from .modules.lavis_model import LavisEncoder, LavisBLIPGraphHead, LavisLorentzBLIPGraphHead 
from transformers import PerceiverConfig
from .modules.utils import freeze_blip, ManifoldMapper 
from .modules.perceiver import MultiModalModel 
from peft import get_peft_model, LoraConfig, TaskType
from .hypCLIP import get_lora_clip
from .hypBLIP import get_lora_blip
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from .baseQueueModel import BaseModelWithQueue
from copy import deepcopy
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from transformers import PerceiverImageProcessor
EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'

        
class PerceiverLavisBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model) -> None:
        super(PerceiverLavisBLIPWithQueue, self).__init__(config)
        projection_dim = 256

        model = get_lora_blip(config=config,model=model) 

        self.num_latents = config.num_latents
        self.vision_model = LavisEncoder(
            config,
            body=model.visual_encoder,
            head=model.vision_proj,
            mapper=None,
            use_normalized=config.normalize_image_embed
        )
        self.text_model = LavisEncoder(
            config,
            body=model.text_encoder,
            head=model.text_proj,
            mapper=None,
            use_normalized=config.normalize_text_embed
        )
      
       
        head_config = PerceiverConfig(
            d_latents=projection_dim, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )

        self.head = MultiModalModel(
            config=head_config, 
            d_vision=768,
            d_text=768,
            d_out=projection_dim,
            num_vision_blocks=1,
            num_text_blocks=1,
            mapper=self.mapper
        ) 

        self._init_queue(config, projection_dim)

    def _init_queue(self, config, ft_out):
        self.vision_model_m= deepcopy(self.vision_model) 
        self.text_model_m= deepcopy(self.text_model) 
        self.head_m = deepcopy(self.head) 
        self.model_pairs = [
            [self.vision_model, self.vision_model_m],
            [self.text_model, self.text_model_m],
            [self.head, self.head_m],
        ]
        self.copy_params()
          # create the queue
        if config.manifold == EUCLID:
            self.register_buffer("image_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("text_queue", torch.randn(self.queue_size, ft_out).T)
            self.image_queue = nn.functional.normalize(self.image_queue.T, dim=-1).T
            self.text_queue = nn.functional.normalize(self.text_queue.T, dim=-1).T
            self.itm_head = DisModel(dim=ft_out, layer_dims=[512, 512, 1])
        else:
            self.register_buffer("image_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.register_buffer("text_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.manifold.assert_check_point_on_manifold(self.image_queue.T)
            self.manifold.assert_check_point_on_manifold(self.text_queue.T)
            self.itm_head = LorentzDisModel(self.manifold, dim=ft_out, layer_dims=[512, 512])

        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def get_attend_mask(self, num_latents):
        zeros = torch.zeros(num_latents)
        hiddens = torch.zeros(num_latents) - float('inf')

        vis_mask = torch.cat([hiddens, zeros]).expand([num_latents, -1])
        text_mask = torch.cat([zeros, hiddens]).expand([num_latents, -1])
        attention_mask = torch.cat([text_mask, vis_mask], dim=0)
        
        return attention_mask.to(self.device)



    def eval(self):
        self.head.eval()
        self.head_m.eval()
        self.text_model.eval()
        self.text_model_m.eval()
        self.vision_model.eval()
        self.vision_model_m.eval()

    def train(self):
        self.head.train()
        self.head_m.train()
        self.text_model.train()
        self.vision_model.train()
        self.text_model_m.train()
        self.vision_model_m.train()
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_id: torch.FloatTensor,
        epoch: int,
        iters: int,
        num_iters_per_epoch:int,
    ):
        idx = image_id

        alpha = self.alpha * self._rampup_factor(
            epoch=epoch,
            iters=iters,
            num_iters_per_epoch=num_iters_per_epoch,
        )

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()
        
        
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_output = self.vision_model(
            pixel_values=pixel_values,
        )
        
        text_embeds = text_output[1]
        image_embeds = image_output[1]

        text_hidden_states = [text_output[0]] 
        image_hidden_states = [image_output[0]] 
        self_attend_mask = self.get_attend_mask(self.num_latents )
        
        itc_text, itc_vision, itm_text, itm_vision = self.head(
            text_ori=text_embeds, 
            vision_ori=image_embeds,
            text_inputs=text_hidden_states, 
            vision_inputs=image_hidden_states, 
            self_attend_mask=self_attend_mask
        ) 

        text_feat = self.postprocess_embeds(itc_text)
        image_feat = self.postprocess_embeds(itc_vision)
        text_feat_itm = self.postprocess_embeds(itm_text)
        image_feat_itm = self.postprocess_embeds(itm_vision)
        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)


        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_output_m = self.vision_model_m(
                pixel_values=pixel_values,
            )
            
            text_output_m = self.text_model_m(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds_m = text_output_m[1]
            image_embeds_m = image_output_m[1]

    
            itc_text_m, itc_vision_m, _, _ = self.head_m(
                text_ori=text_embeds_m, 
                vision_ori= image_embeds_m,
                text_inputs=[text_output_m[0]], 
                vision_inputs= [image_output_m[0]], 
                self_attend_mask=self_attend_mask
            ) 

            text_feat_m = self.postprocess_embeds(itc_text_m)
            image_feat_m = self.postprocess_embeds(itc_vision_m)

            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = self.dist_func(image_feat_m, text_feat_m_all.T) 
            sim_t2i_m = self.dist_func(text_feat_m, image_feat_m_all.T)

            self.manifold.assert_check_point_on_manifold(text_feat_m_all.T)
            self.manifold.assert_check_point_on_manifold(image_feat_m_all.T)
            sim_i2t_targets = alpha * (
                F.softmax(sim_i2t_m * _scale, dim=-1)
            ) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * (
                F.softmax(sim_t2i_m * _scale, dim=-1)
            ) + (1 - alpha) * sim_targets

        sim_i2t = self.dist_func(image_feat, text_feat_m_all.T)     
        sim_t2i = self.dist_func(text_feat, image_feat_m_all.T) 

        margin_loss = self.margin_loss(pos_idx=pos_idx, text_feat=text_feat, image_feat=image_feat, text_world=text_feat_m_all.T, image_world=image_feat_m_all.T)

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t * _scale, dim=1) * sim_i2t_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i * _scale, dim=1) * sim_t2i_targets, dim=-1
        ).mean()      
     

        loss_itc = (loss_i2t + loss_t2i) / 2
        sims = self.dist_func(image_feat, text_feat)
        # loss_itm, itm_acc = self.itm_loss(imgs=image_feat_itm, texts=text_feat_itm, sims_i2t=sims)
        loss_itm, itm_acc = torch.tensor(0.0), torch.tensor(0.0) 

        in_batch_target = torch.arange(bsize).to(self.device)
        
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/itm_loss": loss_itm.item(),
            "logits/margin_loss": margin_loss.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc": (self.dist_func(image_feat, text_feat).argmax(-1) == in_batch_target).float().mean().item(),
            "logits/itm_acc": itm_acc.item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        loss = loss_itc + loss_itm + margin_loss 
        return  loss, stats

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_output = self.text_model(
           input_ids=input_ids, 
           attention_mask=attention_mask, 
        )
        text_embeds = text_output[1]
        text_output = self.head.get_text_features(text_ori=text_embeds, text_inputs=[text_output[0]])
        text_feat = self.postprocess_embeds(text_output)
        return text_feat

    def get_vision_features(self, pixel_values:torch.Tensor):
        # TODO 
        image_output = self.vision_model(
            pixel_values=pixel_values, 
        )
        image_embeds = image_output[1]
        image_output = self.head.get_vision_features(vision_ori=image_embeds, vision_inputs=[image_output[0]])
        image_feat = self.postprocess_embeds(image_output)
        return image_feat





class PerceiverCLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(PerceiverCLIPWithQueue, self).__init__(config)

        text_model = CLIPTextModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        self.vision_model, self.text_model = get_lora_clip(config, vision_model=vision_model, text_model=text_model)
        self.num_latents = config.num_latents
      
       
        head_config = PerceiverConfig(
            d_latents=text_model.config.projection_dim, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )

        self.head = MultiModalModel(
            config=head_config, 
            d_vision=vision_model.config.hidden_size,
            d_text=text_model.config.hidden_size,
            d_out=text_model.config.projection_dim,
            num_vision_blocks=self.config.num_vision_hidden_states,
            num_text_blocks=self.config.num_text_hidden_states,
            mapper=self.mapper
        ) 

        self._init_queue(config, vision_model.config.projection_dim)

    def _init_queue(self, config, ft_out):
        self.vision_model_m= deepcopy(self.vision_model) 
        self.text_model_m= deepcopy(self.text_model) 
        self.head_m = deepcopy(self.head) 
        self.model_pairs = [
            [self.vision_model, self.vision_model_m],
            [self.text_model, self.text_model_m],
            [self.head, self.head_m],
        ]
        self.copy_params()
          # create the queue
        if config.manifold == EUCLID:
            self.register_buffer("image_queue", torch.randn(self.queue_size, ft_out).T)
            self.register_buffer("text_queue", torch.randn(self.queue_size, ft_out).T)
            self.image_queue = nn.functional.normalize(self.image_queue.T, dim=-1).T
            self.text_queue = nn.functional.normalize(self.text_queue.T, dim=-1).T
            self.itm_head = DisModel(dim=ft_out, layer_dims=[512, 512, 1])
        else:
            self.register_buffer("image_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.register_buffer("text_queue", self.manifold.random(self.queue_size, ft_out + 1).T)
            self.manifold.assert_check_point_on_manifold(self.image_queue.T)
            self.manifold.assert_check_point_on_manifold(self.text_queue.T)
            self.itm_head = LorentzDisModel(self.manifold, dim=ft_out, layer_dims=[512, 512])

        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def get_attend_mask(self, num_latents):
        zeros = torch.zeros(num_latents)
        hiddens = torch.zeros(num_latents) - float('inf')

        vis_mask = torch.cat([hiddens, zeros]).expand([num_latents, -1])
        text_mask = torch.cat([zeros, hiddens]).expand([num_latents, -1])
        attention_mask = torch.cat([text_mask, vis_mask], dim=0)
        
        return attention_mask.to(self.device)



    def eval(self):
        self.vision_model.eval()
        self.text_model.eval()
        self.text_model_m.eval()
        self.vision_model_m.eval()
        self.head.eval()
        self.head_m.eval()

    def train(self):
        self.vision_model.train()
        self.text_model.train()
        self.vision_model_m.train()
        self.text_model_m.train()
        self.head.train()
        self.head_m.train()
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_id: torch.FloatTensor,
        epoch: int,
        iters: int,
        num_iters_per_epoch:int,
    ):
        idx = image_id

        alpha = self.alpha * self._rampup_factor(
            epoch=epoch,
            iters=iters,
            num_iters_per_epoch=num_iters_per_epoch,
        )

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()
        
        
        text_output = self.text_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        image_output = self.vision_model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        text_embeds = self.text_model.text_projection(text_output[1])
        image_embeds = self.vision_model.visual_projection(image_output[1])

        text_hidden_states = text_output.hidden_states
        image_hidden_states = image_output.hidden_states
        self_attend_mask = self.get_attend_mask(self.num_latents)
        
        itc_text, itc_vision, itm_text, itm_vision = self.head(
            text_ori=text_embeds, 
            vision_ori=image_embeds,
            text_inputs=text_hidden_states, 
            vision_inputs=image_hidden_states, 
            self_attend_mask=self_attend_mask
        ) 

        text_feat = self.postprocess_embeds(itc_text)
        image_feat = self.postprocess_embeds(itc_vision)
        text_feat_itm = self.postprocess_embeds(itm_text)
        image_feat_itm = self.postprocess_embeds(itm_vision)
        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)


        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_output_m = self.vision_model_m.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            
            text_output_m = self.text_model_m.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_embeds_m = self.text_model_m.text_projection(text_output_m[1])
            image_embeds_m = self.vision_model_m.visual_projection(image_output_m[1])

    
            itc_text_m, itc_vision_m, _, _ = self.head_m(
                text_ori=text_embeds_m, 
                vision_ori= image_embeds_m,
                text_inputs=text_output_m.hidden_states, 
                vision_inputs= image_output_m.hidden_states, 
                self_attend_mask=self_attend_mask
            ) 

            text_feat_m = self.postprocess_embeds(itc_text_m)
            image_feat_m = self.postprocess_embeds(itc_vision_m)

            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = self.dist_func(image_feat_m, text_feat_m_all.T) 
            sim_t2i_m = self.dist_func(text_feat_m, image_feat_m_all.T)

            self.manifold.assert_check_point_on_manifold(text_feat_m_all.T)
            self.manifold.assert_check_point_on_manifold(image_feat_m_all.T)
            sim_i2t_targets = alpha * (
                F.softmax(sim_i2t_m * _scale, dim=-1)
            ) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * (
                F.softmax(sim_t2i_m * _scale, dim=-1)
            ) + (1 - alpha) * sim_targets

        sim_i2t = self.dist_func(image_feat, text_feat_m_all.T)     
        sim_t2i = self.dist_func(text_feat, image_feat_m_all.T) 

        # margin_loss = self.margin_loss(pos_idx=pos_idx, text_feat=text_feat_itm, image_feat=image_feat_itm, text_world=text_feat_m_all.T, image_world=image_feat_m_all.T)
        margin_loss = self.margin_loss(pos_idx=torch.eye(text_feat_itm.shape[0]).to(self.device), text_feat=text_feat, image_feat=image_feat, text_world=text_feat_itm, image_world=image_feat_itm)

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t * _scale, dim=1) * sim_i2t_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i * _scale, dim=1) * sim_t2i_targets, dim=-1
        ).mean()      
     

        loss_itc = (loss_i2t + loss_t2i) / 2
        sims = self.dist_func(image_feat, text_feat)
        loss_itm, itm_acc = self.itm_loss(imgs=image_feat_itm, texts=text_feat_itm, sims_i2t=sims)
        # loss_itm, itm_acc = torch.tensor(0.0), torch.tensor(0.0) 

        in_batch_target = torch.arange(bsize).to(self.device)
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/itm_loss": loss_itm.item(),
            "logits/margin_loss": margin_loss.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc": (self.dist_func(image_feat, text_feat).argmax(-1) == in_batch_target).float().mean().item(),
            "logits/itm_acc": itm_acc.item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        loss = loss_itc + loss_itm + margin_loss 
        return  loss, stats

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_output = self.text_model.text_model(
           input_ids=input_ids, 
           attention_mask=attention_mask, 
           output_hidden_states=True
        )
        text_embeds = self.text_model.text_projection(text_output[1])
        text_output = self.head.get_text_features(text_ori=text_embeds, text_inputs=text_output.hidden_states)
        text_feat = self.postprocess_embeds(text_output)
        return text_feat

    def get_vision_features(self, pixel_values:torch.Tensor):
        # TODO 
        image_output = self.vision_model.vision_model(
            pixel_values=pixel_values, 
            output_hidden_states=True
        )
        image_embeds = self.vision_model.visual_projection(image_output[1])
        image_output = self.head.get_vision_features(vision_ori=image_embeds, vision_inputs=image_output.hidden_states)
        image_feat = self.postprocess_embeds(image_output)
        return image_feat


