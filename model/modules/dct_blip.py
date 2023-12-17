
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
    BlipModel
)
import math
from transformers.models.blip.modeling_blip import BlipImageTextMatchingModelOutput
from lavis import BlipRetrieval
from transformers import CLIPVisionModel, AutoModel 
from .dct import dc_transform, dct, idct


class CompressedModela(nn.Module):
    def __init__(self, model):
        self.vision_model = None
        self.text_model = None
        self.vision_proj = None
        self.text_proj = None 
    
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

    def dc_transform(self, x, use_reconstucted_freq=False):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        # dct along T dimension
        T, B, C = x_dct.size()
        x_dct_mean = torch.abs(x_dct.permute(1,0,2).mean(0).mean(1))
        threshold = torch.abs(torch.quantile(x_dct_mean, 0.8, dim=-1, keepdim=True) )
        indices = torch.where(x_dct_mean > threshold)
        last_index = indices[0][-1].item() if indices[0].numel() > 0 else -1
        # feel free to play with any method here
        if use_reconstucted_freq:
       
            x_dct = x_dct[:last_index, :, :]
   
        return idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half).permute(1,0,2), x_dct.permute(1,0,2)
    
    def svd_transform(self, x, use_reconstucted_freq=False):
        x = x.type(torch.float32)
        u, s, v = torch.svd(x.to(torch.float32))
        threshold = torch.quantile(torch.log(s), 0.1, dim=-1, keepdim=True)
        k = torch.where(torch.log(s) > threshold)[-1]
        if use_reconstucted_freq:
            x = torch.matmul(torch.matmul(u[:, :k, :k], torch.diag_embed(s[:,:k])), v.mT[:,:k, :])
        return x.type(torch.half), s

    def get_vision_features(self, pixel_values, apply_fourier=True):
        raise NotImplementedError("This method is not implemented yet")


    def get_text_features(self, input_ids, attention_mask):
        raise NotImplementedError("This method is not implemented yet")
    
class DCTHFBlip(nn.Module):
    config_class = BlipConfig

    def __init__(self, model:AutoModel):
        super().__init__()

        self.vision_model = model.vision_model

        self.text_encoder = model.text_model 

        self.vision_proj = model.visual_projection 

        self.text_proj = model.text_projection 
        self.r_list = nn.ParameterList([
            1.0, 0.9, 1.0, 1.0, 0.9, 1.0,
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



class DCTLAVISBlip(nn.Module):
    config_class = BlipConfig

    def __init__(self, model:BlipRetrieval, r_list=None):
        super().__init__()

        self.vision_model = model.visual_encoder

        self.text_encoder = model.text_encoder 

        self.vision_proj = model.vision_proj 

        self.text_proj = model.text_proj 

        self.r_list = [0.98] * int(len(self.vision_model.blocks)/2)
        self.r_list += [1.0] * int(len(self.vision_model.blocks)/2)
        
        
        final_len = 576

        self.freq_scaler = nn.Parameter(torch.tensor(1.0))
        # self.freq_bias = nn.Parameter(torch.zeros(1, final_len, 768))
        for r in self.r_list:
            final_len = math.ceil(final_len * r)



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
    

    def dc_transform(self, x, use_reconstucted_freq=False):
        # cufft doesn't accept fp16
        # dct along T dimension
        # x = x.type(torch.float32).permute(1,0,2)
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        x_dct_mean = torch.abs(x_dct.permute(1,0,2).mean(0).mean(1))
        # print('got here')

        threshold = torch.abs(torch.quantile(x_dct_mean, 0.8, dim=-1, keepdim=True) )
        indices = torch.where(x_dct_mean > threshold)
        last_index = indices[0][-1].item() if indices[0].numel() > 0 else -1
        # feel free to play with any method here
        if use_reconstucted_freq:
            # x_dct = x_dct[:math.ceil(T * r), :, :]
            x_dct = x_dct[:last_index, :, :]
   
        return idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half).permute(1,0,2), x_dct.permute(1,0,2)


       
    def get_vision_features(self, pixel_values, apply_fourier=True, return_all_fourier_signals=False):
        B = pixel_values.shape[0]
        x = self.vision_model.patch_embed(pixel_values)
        hidden_states = []
        dct_signals = [] 

        cls_tokens = self.vision_model.cls_token.expand(
            B, -1, -1
        ) 
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.vision_model.pos_embed[:, : x.size(1), :]
        x = self.vision_model.pos_drop(x)

        for i, blk in enumerate(self.vision_model.blocks):
         
            if i > len(self.vision_model.blocks)/2 :
                cls = x[:, 0, :].unsqueeze(1)
                # print(x.shape)
                state, dct_signal = self.dc_transform(
                    x[:, 1:, :], 
                    use_reconstucted_freq=apply_fourier
                )
                # print(state.shape)
                x = torch.cat([cls, state], dim=1)
                if return_all_fourier_signals or i == len(self.vision_model.blocks)-1:
                    dct_signals.append(dct_signal)
                    hidden_states.append(state)

            x = blk(x)
        x = self.vision_model.norm(x)

        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed, hidden_states, dct_signals

    def get_text_features(self, input_ids, attention_mask):
        with torch.no_grad():
            class Text(object):
                pass
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            question_embeds = self.text_encoder.forward_text(text)
            last_hidden_state = question_embeds[0] 
            text_embed = self.text_proj(last_hidden_state[:,0,:])

            return  last_hidden_state, text_embed


class DCTHFClip(nn.Module):

    def __init__(self, model:AutoModel):
        super().__init__()

        self.vision_model = model.vision_model

        self.text_model = model.text_model 

        self.vision_proj = model.visual_projection 

        self.text_proj = model.text_projection 
        ori_len = 576
        final_len = 576

        self.r_list = [1.0] * int(len(self.vision_model.encoder.layers)/2)
        self.r_list.extend([0.9] * int(len(self.vision_model.encoder.layers)/2))
        for r in self.r_list:
            final_len = math.ceil(final_len * r)


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

    def dc_transform(self, x, use_reconstucted_freq=True):
        # cufft doesn't accept fp16
        # dct along T dimension
        # x = x.type(torch.float32).permute(1,0,2)
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        if use_reconstucted_freq:
            x_dct_mean = torch.abs(x_dct.permute(1,0,2).mean(0).mean(1))
            threshold =torch.quantile(x_dct_mean, 0.7, dim=-1, keepdim=True, interpolation='linear') 
            indices = torch.where(x_dct_mean > threshold)
            last_index = indices[0][-1].item() if indices[0].numel() > 0 else -1
            x_dct = x_dct[:last_index:, :]
     

        return idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half).permute(1,0,2), x_dct.permute(1,0,2)
       
    
    def get_vision_features(self, pixel_values, apply_fourier=True, return_all_fourier_signals=False):
        dct_signals = []
        all_hidden_states = []
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i > len(self.vision_model.encoder.layers)/2 :
           
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, dct_signal = self.dc_transform(
                    hidden_states[:, 1:, :], 
                    use_reconstucted_freq=apply_fourier
                )
                hidden_states = torch.cat([cls, state], dim=1)
                if return_all_fourier_signals or i == len(self.vision_model.encoder.layers)-1:
                    dct_signals.append(dct_signal)
                    all_hidden_states.append(hidden_states)
            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        vision_embed = self.vision_proj(pooled_output)
        

        return hidden_states, vision_embed, all_hidden_states, dct_signals

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs[1]
        text_embed = self.text_proj(pooled_output)

        return  text_outputs[0], text_embed