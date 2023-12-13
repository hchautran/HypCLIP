
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
    BlipModel
)
from transformers.models.blip.modeling_blip import BlipImageTextMatchingModelOutput
from lavis import BlipRetrieval
from transformers import CLIPVisionModel, CLIPModel
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

        


class DCTBlip(nn.Module):
    config_class = BlipConfig

    def __init__(self, model:BlipModel):
        super().__init__()

        self.vision_model = model.vision_model

        self.text_encoder = model.text_encoder 

        self.vision_proj = model.vision_proj 

        self.text_proj = model.text_proj 
        self.r_list = nn.ParameterList([
            1.0, 0.7, 1.0, 1.0, 0.7, 1.0,
            1.0, 0.9, 1.0, 1.0, 1.0, 1.0,
        ])


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
            return self.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)

       
    
    def get_vision_features(self, pixel_values, apply_fourier=True):
        state = self.vision_model.embeddings(pixel_values)
        # state = self.vision_model.pre_layrnorm(state)
        hidden_states = []
        hidden_states.append(state)
        r=1.0

        for i, layer in enumerate(self.vision_model.encoder.layers):
            state = layer(state, None, None)[0]
            cls = state[:, 0, :].unsqueeze(1)
            state = dc_transform(state[:,1:,:].permute(1,0,2), r=(self.r_list[i] if (self.training or apply_fourier) else 1.0)).permute(1,0,2)
            state = torch.cat([cls, state], dim=1)
            # print(self.r_list[i])
            # print(state.shape)
                 
            
            hidden_states.append(state)
        # print('---'*50)

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


class DCTLAVISBlip(nn.Module):
    config_class = BlipConfig

    def __init__(self, model:BlipRetrieval):
        super().__init__()

        self.vision_model = model.visual_encoder

        self.text_encoder = model.text_encoder 

        self.vision_proj = model.vision_proj 

        self.text_proj = model.text_proj 
        self.r_list = nn.ParameterList([
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
            0.9, 0.9, 0.9, 0.9, 1.0, 1.0,
        ])


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
            return self.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)

       
    def get_vision_features(self, pixel_values, apply_fourier=True):
        # outputs = self.vision_model.forward_features(
        #     pixel_values,
        # )
        # last_hidden_state = outputs
        # pooled_output = last_hidden_state[:, 0, :]
        B = pixel_values.shape[0]
        x = self.vision_model.patch_embed(pixel_values)

        cls_tokens = self.vision_model.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.vision_model.pos_embed[:, : x.size(1), :]
        x = self.vision_model.pos_drop(x)

        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
            cls = x[:, 0, :].unsqueeze(1)
            state = dc_transform(x[:,1:,:].permute(1,0,2), r=(self.r_list[i] if (self.training or apply_fourier) else 1.0)).permute(1,0,2)
            x = torch.cat([cls, state], dim=1)
        x = self.vision_model.norm(x)

        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed

    def get_text_features(self, input_ids, attention_mask):
        class Text(object):
            pass
        text = Text() 
        text.input_ids=input_ids
        text.attention_mask=attention_mask
        question_embeds = self.text_encoder.forward_text(text)
        last_hidden_state = question_embeds[0] 
        text_embed = self.text_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed


class DCTClip(nn.Module):

    def __init__(self, model:CLIPModel):
        super().__init__()

        self.vision_model = model.vision_model

        self.text_model = model.text_model 

        self.vision_proj = model.visual_projection 

        self.text_proj = model.text_projection 

        self.r_list = nn.ParameterList([
            1.0, 0.7, 1.0, 1.0, 0.7, 1.0,
            1.0, 0.9, 1.0, 1.0, 1.0, 1.0,
        ])


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
            return self.get_vision_features(pixel_values=pixel_values, apply_fourier=apply_fourier)

       
    
    def get_vision_features(self, pixel_values, apply_fourier=True):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        for layer in self.vision_model.encoder.layers:
            hidden_states = layer(
                hidden_states,
            )[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        vision_embed = self.vision_proj(pooled_output)
        

        return last_hidden_state, vision_embed

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs[1]
        text_embed = self.text_proj(pooled_output)

        return  text_outputs[0], text_embed