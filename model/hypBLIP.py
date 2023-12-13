import torch
import torch.nn as nn
from .modules.model import BLIPEncoder
from transformers import BlipForImageTextRetrieval
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel 
from model.baseQueueModel import BaseModelWithQueue 
from model.baseDistilledModel import BaseModel as BaseDistilModel 
from .modules.utils import ManifoldMapper
from peft import get_peft_model, LoraConfig, TaskType
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
        lora_dropout=0.2, 
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config) 
    model.print_trainable_parameters()
    return model



class HypBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config) -> None:
        super(HypBLIPWithQueue, self).__init__(config)
        self.config = config

        model = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        model = get_lora_blip(config, model)
      
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = model.text_proj
        vision_head = model.vision_proj
        mapper = None
   
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.vision_model = BLIPEncoder(
            config,
            body=vision_body,
            head=vision_head,
            manifold_mapper=mapper
        )
        self.text_model = BLIPEncoder(
            config,
            body=text_body,
            head=text_head,
            manifold_mapper=mapper
        )
        self._init_queue(config, 256)

class LavisBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIPWithQueue, self).__init__(config)
        
    
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.model = LavisEncoder(
            config,
            vision_body=model.visual_encoder,
            vision_head=model.vision_proj,
            text_body=model.text_encoder,
            text_head=model.text_proj,
            mapper=mapper,
            use_normalized=config.normalize_image_embed
        )
        self._init_queue(config, 256)
    

class LavisBLIP(BaseModel):
    def __init__(self, config, model:BlipRetrieval) -> None:
        super(LavisBLIP, self).__init__(config)
        
    
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if self.config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)

        self.model = LavisEncoder(
            config,
            vision_body=model.visual_encoder,
            vision_head=model.vision_proj,
            text_body=model.text_encoder,
            text_head=model.text_proj,
            mapper=mapper,
            use_normalized=config.normalize_image_embed
        )

 


class LavisHypGraphBLIP(BaseModel):
    def __init__(self, config, model) -> None:
        super(LavisHypGraphBLIP, self).__init__(config)
        
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
            self.model = LavisLorentzBLIPGraphModel(
                manifold=self.manifold,
                d_text=768,
                d_vision=768,
                ft_out=256,
                config=config,
                text_body=model.text_encoder,
                text_head=model.text_proj,
                vision_body=model.visual_encoder,
                vision_head=model.vision_proj,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )
            
        else:
            self.model = LavisBLIPGraphModel(
                d_text=768,
                d_vision=768,
                ft_out=256,
                config=config,
                text_body=model.text_encoder,
                text_head=model.text_proj,
                vision_body=model.visual_encoder,
                vision_head=model.vision_proj,
                manifold_mapper=mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                num_hidden_layers=config.num_proj_layers,
                use_root=config.use_root
            )
    

        # self.eu_logit_scale = model.temp
        # self.logit_scale = model.temp
        
class LavisHypGraphBLIPWithQueue(BaseModelWithQueue):
    def __init__(self, config, model) -> None:
        super(LavisHypGraphBLIPWithQueue, self).__init__(config)
        
        model = get_lora_lavis_blip(config=config,model=model) 
        self.config = config
        mapper = None
        if config.manifold != EUCLID:
            mapper = ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
            self.model = LavisLorentzBLIPGraphModel(
                manifold=self.manifold,
                d_text=768,
                d_vision=768,
                ft_out=256,
                config=config,
                text_body=model.text_encoder,
                text_head=model.text_proj,
                vision_body=model.visual_encoder,
                vision_head=model.vision_proj,
                itm_head=model.itm_head,
                manifold_mapper=mapper,
            )
            
        else:
            self.model = LavisBLIPGraphModel(
                d_text=768,
                d_vision=768,
                ft_out=256,
                config=config,
                text_body=model.text_encoder,
                text_head=model.text_proj,
                vision_body=model.visual_encoder,
                vision_head=model.vision_proj,
            )

        self.tokenizer = model.tokenizer
        self.itm_model = deepcopy(model.text_encoder)
        self.itm_head = model.itm_head
        self._init_queue(config, 256)
    
    def distil_itm_loss(self, idx, input_ids, attention_mask, pixel_values, image_atts, image_embeds, text_hidden_states, image_hidden_states, sim_t2i, sim_i2t):
        encoder_input_ids = input_ids.clone()
        T=2
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        bs = pixel_values.size(0)
       
        with torch.no_grad():
            output_pos = self.itm_model(
                encoder_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            mask = torch.eq(idx, idx.t())

            sim_i2t = sim_i2t / self.logit_scale
            sim_t2i = sim_t2i / self.logit_scale

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

            # select a negative image (from same rank) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from same rank) for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(attention_mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([attention_mask, text_atts_neg], dim=0)

            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)

            output_neg = self.itm_model(
                text_ids_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
            )

            vl_embeddings = torch.cat(
                [
                    output_pos.last_hidden_state[:, 0, :],
                    output_neg.last_hidden_state[:, 0, :],
                ],
                dim=0,
            )
            teacher_logits = self.itm_head(vl_embeddings)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(self.device)
        
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_hidden_states[neg_idx])

        # select a negative text for each image
        text_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_hidden_states[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        text_hidden_states = torch.cat(
            [text_hidden_states, text_hidden_states, text_ids_neg], dim=0
        )  # pos, pos, neg

        image_hidden_states = torch.cat(
            [image_hidden_states, image_embeds_neg, image_hidden_states], dim=0
        )  # pos, neg, pos


        student_logits = self.model.compute_itm(
            text_latents=text_hidden_states, 
            vision_latents=image_hidden_states, 
        ) 
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

        
        itm_acc = (student_logits.argmax(-1) == itm_labels).float().sum()/itm_labels.shape[0]
        label_loss = self.itm_criterion(student_logits, itm_labels)
        print(soft_targets_loss.item(), label_loss.item())
        itm_loss = 0.2 * soft_targets_loss +  label_loss
        return itm_loss, itm_acc
    
    
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
        
        text_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_output = self.model(
            pixel_values=pixel_values
        )
        text_embeds = text_output[1] 
        image_embeds = image_output[1] 
        text_hidden_states = text_output[0]
        vision_hidden_states = image_output[0]

        text_feat = self.postprocess_embeds(text_embeds)
        image_feat = self.postprocess_embeds(image_embeds)
        self.manifold.assert_check_point_on_manifold(text_feat)
        self.manifold.assert_check_point_on_manifold(image_feat)
        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        with torch.no_grad():
            self.logit_scale.clamp_(0.001, 0.5)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.model_m(
                pixel_values=pixel_values 
            )

            text_embeds_m = self.model_m(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            image_feat_m = self.postprocess_embeds(image_embeds_m[1])
            text_feat_m = self.postprocess_embeds(text_embeds_m[1])

            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = self.dist_func(image_feat_m, text_feat_m_all.T) 
            sim_t2i_m = self.dist_func(text_feat_m, image_feat_m_all.T)
         
            sim_i2t_targets = alpha * (
                F.softmax(sim_i2t_m / self.logit_scale, dim=-1)
            ) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * (
                F.softmax(sim_t2i_m / self.logit_scale, dim=-1)
            ) + (1 - alpha) * sim_targets
           



            self.manifold.assert_check_point_on_manifold(text_feat_m_all.T)
            self.manifold.assert_check_point_on_manifold(image_feat_m_all.T)

        sim_i2t = self.dist_func(image_feat, text_feat_m_all.T) 
        sim_t2i = self.dist_func(text_feat, image_feat_m_all.T)
      

        # if epoch >= 0:
        margin_loss  = self.margin_loss(pos_idx=pos_idx, text_feat=text_feat, image_feat=image_feat, text_world=text_feat_m_all.T, image_world=image_feat_m_all.T)
        # else:
            # margin_loss  = torch.tensor(0.0) 

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t / self.logit_scale, dim=1) * sim_i2t_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i / self.logit_scale, dim=1) * sim_t2i_targets, dim=-1
        ).mean()      
    
       
        loss_itc = self.config.weight_i2t * loss_i2t + (1-self.config.weight_i2t) * loss_t2i
        image_atts = torch.ones(vision_hidden_states.size()[:-1], dtype=torch.long).to(
            self.device
        )
      

        sims = self.dist_func(image_feat, text_feat)
        loss_itm, itm_acc = self.distil_itm_loss(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_atts=image_atts,
            image_embeds=vision_hidden_states,
            text_hidden_states=text_hidden_states, 
            image_hidden_states=vision_hidden_states, 
            sim_i2t=sims, 
            sim_t2i=sims.T
        )
        

        in_batch_target = torch.arange(bsize).to(self.device)
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/itm_loss": loss_itm.item(),
            "logits/margin_loss": margin_loss.item(),
            "logits/min": sims.min().item(),
            "logits/mean": sims.mean().item(),
            "logits/max": sims.max().item(),
            "logits/acc": (sims.argmax(-1) == in_batch_target).float().mean().item(),
            "logits/itm_acc": itm_acc.item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        loss = loss_itc + loss_itm + margin_loss 
        return  loss, stats



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
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
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
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
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
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
            self.text_model = GraphModel(
                ft_in=model.config.text_config.hidden_size,
                ft_out=model.config.image_text_hidden_size,
                config=config,
                body=text_body,
                head=text_head,
                manifold_mapper=self.mapper,
                num_layers=1,
                hidden_size=config.proj_layer_hidden_sizes,
                graph_hidden_channels=config.graph_hidden_channels,
                num_hidden_layers=config.num_proj_layers,
            )
        
        self._init_queue(config, model.config.image_text_hidden_size)




