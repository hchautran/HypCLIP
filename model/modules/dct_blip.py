
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipPreTrainedModel, 
    BlipConfig, 
    BlipVisionModel, 
    BlipTextModel,
    BlipModel
)
from transformers.models.blip.modeling_blip import BlipImageTextMatchingModelOutput
from .dct import dc_transform


class BLIPEncoder(nn.Module): 
    def __init__(self, text_body, text_head, vision_body, vision_head) -> None:
        super().__init__()
        self.text_body = text_body 
        self.text_head = text_head
        self.vision_body = vision_body 
        self.vision_head = vision_head
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.vision_body(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=True
            )
            pooled_output = outputs[0][:,0,:]
            final_feat = self.vision_head(pooled_output) 
        else:
            outputs = self.text_body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            pooled_output = outputs[0][:,0,:]
            final_feat = self.text_head(pooled_output) 

      
        return outputs[0], final_feat 

class DCTBlipForImageTextRetrieval(nn.Module):
    config_class = BlipConfig

    def __init__(self, model:BlipModel):
        super().__init__()

        self.vision_model = model.vision_model

        self.text_encoder = model.text_encoder 

        self.vision_proj = model.vision_proj 

        self.text_proj = model.text_proj 



    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.LongTensor] = None,
        apply_fourier: Optional[torch.LongTensor] = True,
        
    ):
        if input_ids is not None:
            return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.get_vision_features(pixel_values=pixel_values, apply_fourier=True)

       
    
    def get_vision_features(self, pixel_values, apply_fourier=True):
        state = self.vision_model.embeddings(pixel_values)
        # state = self.vision_model.pre_layrnorm(state)
        hidden_states = []
        hidden_states.append(state)

        for i, layer in enumerate(self.vision_model.encoder.layers):
            state = layer(state, None, None)[0]
            cls = state[:, 0, :].unsqueeze(1)
            if i > 6:
                state = dc_transform(state[:,1:,:].permute(1,0,2), r=(0.7 if (self.training or apply_fourier) else 1.0)).permute(1,0,2)
                state = torch.cat([cls, state], dim=1)
            
            hidden_states.append(state)

        last_hidden_state = self.vision_model.post_layernorm(state)
        pooled_output = last_hidden_state[:, 0, :]
        vision_embed = self.vision_proj(pooled_output)
        return state, vision_embed

    def get_text_features(self, input_ids, attention_mask):
        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = question_embeds[0] 
        text_embed = self.text_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed
