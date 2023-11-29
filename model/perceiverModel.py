import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
import torch.nn.functional as F
from .modules.blip import LavisEncoder
from transformers import PerceiverConfig
from .modules.perceiver import MultiModalModel 
from .hypCLIP import get_lora_clip
from .hypBLIP import get_lora_blip
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from .baseQueueModel import BaseModelWithQueue
from copy import deepcopy
from .modules.discriminator import Discriminator as DisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from hyptorch.lorentz.layers import LorentzMLR, LorentzLinear 
EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'

        



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
        self.temp = nn.Parameter(0.07 * torch.ones([]))


       
        head_config = PerceiverConfig(
            d_latents=config.d_latents, 
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
            mapper=self.mapper,
            num_blocks=2
        ) 
        self.text_proj = text_model.text_projection
        self.vision_proj = vision_model.visual_projection
        self.dropout=nn.Dropout1d(p=0.1)
        self.perceiver_vision_proj = nn.Linear(config.d_latents, text_model.config.projection_dim) 

        if config.manifold == EUCLID:
            self.itm_head = nn.Linear(config.d_latents, 2)
        else:
            self.itm_head = LorentzMLR(self.manifold, config.d_latents+1, 2)


    def get_sim_i2t(self, q:torch.Tensor, t:torch.Tensor ):
        if self.config.manifold == EUCLID:
            sim_q2t = torch.matmul(
                q.unsqueeze(1), t.unsqueeze(-1)
            ).squeeze()
        else: 
            sim_q2t = -self.manifold.dist_batch(
                q.unsqueeze(1), t.unsqueeze(-1)
            ).squeeze()
        sim_i2t = sim_q2t.mean(-1)
        # sim_i2t, _ = sim_q2t.max(-1)
        return sim_i2t
        
    def get_sim_t2i(self, t:torch.Tensor, q:torch.Tensor):
        if self.config.manifold == EUCLID:
            sim_t2q = torch.matmul(
                t.unsqueeze(1).unsqueeze(1), q.permute(0, 2, 1)
            ).squeeze()
        else: 
            sim_t2q = -self.manifold.dist_batch(
                t.unsqueeze(1).unsqueeze(1), q.permute(0, 2, 1)
            ).squeeze()
        sim_t2i = sim_t2q.mean(-1)
        # sim_t2i, _ = sim_t2q.max(-1)
        return sim_t2i

    
    
    def get_attend_mask(self, num_latents):
        zeros = torch.zeros(num_latents)
        hiddens = torch.zeros(num_latents) - float('inf')

        vis_mask = torch.cat([hiddens, zeros]).expand([num_latents, -1])
        text_mask = torch.cat([zeros, hiddens]).expand([num_latents, -1])
        attention_mask = torch.cat([text_mask, vis_mask], dim=0)
        
        return attention_mask.to(self.device)
    
    def itm_loss(self, text_hidden_states, image_hidden_states, sim_t2i, sim_i2t):
        bs = text_hidden_states.shape[0]
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            mask = (torch.eye(bs) > 0).to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_hidden_states[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_hidden_states[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)

        text_hidden_states = torch.cat(
            [text_hidden_states, text_hidden_states, text_ids_neg], dim=0
        )  # pos, pos, neg

        image_hidden_states = torch.cat(
            [image_hidden_states, image_embeds_neg, image_hidden_states], dim=0
        )  # pos, neg, pos


        itm_text, itm_vision = self.head(
            text_inputs=text_hidden_states, 
            vision_inputs=image_hidden_states, 
            self_attend_mask=None,
            
        ) 
        itm_score = self.itm_head(torch.cat([itm_text, itm_vision], dim=1))


        logits = itm_score.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)

        itm_acc = (logits.argmax(-1) == itm_labels).float().sum()/itm_labels.shape[0]
        loss_itm = F.cross_entropy(logits, itm_labels)
        return loss_itm, itm_acc
        
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_id: torch.FloatTensor,
    ):
        idx = image_id

        with torch.no_grad():
            self.temp.clamp_(0.05, 0.1)

            image_output = self.vision_model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
  
        text_output = self.text_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        text_hidden_states = torch.cat(text_output.hidden_states[(len(text_output.hidden_states) - self.config.num_text_hidden_states):], dim=1)
        text_feat = self.text_proj(text_output[1])

        image_hidden_states = torch.cat(image_output.hidden_states[(len(image_output.hidden_states) - self.config.num_vision_hidden_states):], dim=1)
        image_ori = self.vision_proj(image_output[1])

        
        itc_vision = self.head.get_vision_features(vision_inputs=image_hidden_states, vision_ori=None)

        itc_vision =self.dropout(itc_vision)
        image_feat = self.perceiver_vision_proj(itc_vision)
        image_feat = image_ori.expand(itc_vision.shape[1], -1 ,-1).permute(1,0,2) + image_feat

        text_feat = self.postprocess_embeds(text_feat)
        image_feat = self.postprocess_embeds(image_feat)
        bsize = text_feat.shape[0]
        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        target = torch.arange(bsize).to(self.device)

        sim_i2t = self.get_sim_i2t(image_feat, text_feat)     
        sim_t2i = self.get_sim_t2i(text_feat, image_feat) 
        loss_itc= (
            F.cross_entropy(sim_i2t/self.temp, target, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i/self.temp, target, label_smoothing=0.1)
        ) / 2

        loss_itm, itm_acc = self.itm_loss(
            text_hidden_states=text_hidden_states, 
            image_hidden_states=image_hidden_states, 
            sim_i2t=sim_i2t,
            sim_t2i=sim_t2i
        )

        in_batch_target = torch.arange(bsize).to(self.device)
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/itm_loss": loss_itm.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc_i2t": (self.get_sim_i2t(image_feat, text_feat).argmax(-1) == in_batch_target).float().mean().item(),
            "logits/acc_t2i": (self.get_sim_t2i(text_feat, image_feat).argmax(-1) == in_batch_target).float().mean().item(),
            "logits/itm_acc": itm_acc.item(),
            "logits/curvature": self.manifold.k.item() if self.config.manifold != EUCLID else 0.0 
        }

        loss = loss_itc + loss_itm 
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

        text_feat = self.text_proj(text_output[1])
        text_inputs = torch.cat(text_output.hidden_states, dim=1)
        text_feat = self.postprocess_embeds(text_feat)
        return text_feat, text_inputs

    def get_text_features(self, pixel_values:torch.Tensor):
        image_output = self.vision_model.vision_model(
            pixel_values=pixel_values, 
            output_hidden_states=True
        )
        image_ori = self.vision_proj(image_output[1])
        vision_inputs = torch.cat(image_output.hidden_states, dim=1)
        itc_vision = self.head.get_vision_features(vision_inputs=vision_inputs, vision_ori=None)
        image_feat = self.perceiver_vision_proj(itc_vision)
        image_feat =  image_ori.expand(itc_vision.shape[1], -1 ,-1).permute(1,0,2) + image_feat
        image_feat = self.postprocess_embeds(image_feat)
        return image_feat, vision_inputs
    


