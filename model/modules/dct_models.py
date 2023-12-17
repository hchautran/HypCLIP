
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
from .dct import dct, idct
import torch.fft as fft
import pywt


class CompressedModel(nn.Module):
    def __init__(self, compress_method='dct'):
        super().__init__()
        self.compress_method = compress_method
    
    def find_spans(self, array, threshold=10):
        spans = []
        final_spans = []
        start = array[0]
        current_span = [start]

        for value in array[1:]:
            if value - start <= threshold:
                # Expand the current span
                current_span.append(value)
            else:
                # Start a new span
                spans.append((current_span[0], current_span[-1]))
                current_span = [value]
                start = value

        # Add the last span
        spans.append((current_span[0], current_span[-1]))
        begin = spans[0][0]
        end = spans[0][1]
        for i in range(len(spans) -1):
            if spans[i][0] - end <= threshold:
                end = spans[i][1]
            
            else:
                final_spans.append([begin-5, end+5])
                begin = spans[i][1]
                end = spans[i+1][0]
                
        final_spans.append([begin-5, end+5])
        final_spans[0][0] = 0
        return final_spans

    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_compressed_hidden_state: Optional[torch.LongTensor] = True,
        
    ):
        if input_ids is not None:
            return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.get_vision_features(pixel_values=pixel_values, use_compressed_hidden_state=use_compressed_hidden_state)

    def dc_transform(self, x, use_reconstucted_state=False):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        x_dct_mean = torch.abs(x_dct.permute(1,0,2).mean(0).mean(1))
        threshold = torch.abs(torch.quantile(x_dct_mean, 0.8, dim=-1, keepdim=True) )
        indices = torch.where(x_dct_mean > threshold)
        k = indices[0][-1].item() if indices[0].numel() > 0 else -1
        # k = math.ceil(0.9 * T)
        # feel free to play with any method here
        if use_reconstucted_state:
            x_dct = x_dct[:k, :, :]
            x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half)
   
        return x.permute(1,0,2), x_dct.permute(1,0,2)

    
    def wv_transform(self, x, use_reconstucted_state = False):
        wavelet = 'db3'  # Daubechies wavelet with one vanishing moment
        level = 0
        coeffs = pywt.wavedec(x.cpu().detach(), wavelet, level=level, axis=1)
        # k = math.ceil(0.90 * x.shape[1])
        coeffs_mean = torch.abs(torch.from_numpy(coeffs[0])).mean(0).mean(1)
        threshold = coeffs_mean.mean()
        indices = torch.where(coeffs_mean > threshold)
        k_end = indices[0][-1].item() if indices[0].numel() > 0 else -1
        k_start = indices[0][0].item() if indices[0].numel() > 0 else -1
        if use_reconstucted_state:
            coeffs[0] = coeffs[0][:, k_start:k_end, :]
            x = torch.from_numpy(pywt.waverec(coeffs, wavelet)).to(x.device)
   
        return x, torch.from_numpy(coeffs[0])
    
    def direct(self, x, use_reconstucted_state = False):
        k = math.ceil(0.90 * x.shape[1])
        if use_reconstucted_state:
            x = x[:,:k,:]  
        return x, x
    
    def std_based_compress(self, x, use_reconstucted_state = False):
        threshold = torch.abs(torch.quantile(x.mean(0).std(1), 0.90, dim=-1, keepdim=True) )
        mask = torch.nonzero(torch.abs(x.mean(0).std(1)) > threshold).squeeze()
        if mask.ndimension() == 0: return x, x
        spans  = self.find_spans(mask)

        if use_reconstucted_state:
            states = []
            for span in spans:
                states.append(x[:,span[0]:span[1],:])
            x = torch.cat(states, dim=1)
            
        return x, x

   

    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_fourier_signals=False):
        raise NotImplementedError("This method is not implemented yet")

    def get_text_features(self, input_ids, attention_mask):
        raise NotImplementedError("This method is not implemented yet")
    
    def compress_hidden_state(self, x, use_compressed_hidden_state=False):
        if self.compress_method == 'wv':
            x_reconstructed, energy = self.wv_transform(x ,use_compressed_hidden_state) 
        elif self.compress_method == 'dct':
            x_reconstructed, energy = self.dc_transform(x ,use_compressed_hidden_state) 
        elif self.compress_method == 'std':
            x_reconstructed, energy = self.std_based_compress(x ,use_compressed_hidden_state) 
        else:
            x_reconstructed, energy = self.direct(x ,use_compressed_hidden_state) 
        return  x_reconstructed, energy
    
    


    
class DCTHFBlip(CompressedModel):
    config_class = BlipConfig

    def __init__(self, model:AutoModel, compress_method='dct'):
        super(DCTHFBlip, self).__init__(compress_method)
        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [6, 7]
     

    
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_fourier_signals=False):
        hidden_states = self.vision_model.embeddings(pixel_values)
        all_hidden_states = []
        energy = []

        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:    
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state
                )
                hidden_states = torch.cat([cls, state], dim=1)
                if return_all_fourier_signals or i == len(self.vision_model.encoder.layers)-1:
                    energy.append(cur_energy)
                    all_hidden_states.append(hidden_states)
                # print(hidden_states.shape)

            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]


        last_hidden_state = self.vision_model.post_layernorm(state)
        pooled_output = last_hidden_state[:, 0, :]
        vision_embed = self.vision_proj(pooled_output)
       
        return hidden_states, vision_embed, all_hidden_states, energy

    def get_text_features(self, input_ids, attention_mask):
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = text_output[0] 
        text_embed = self.text_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed



class DCTLAVISBlip(CompressedModel):

    def __init__(self, model:BlipRetrieval, compress_method='dct'):
        super(DCTLAVISBlip, self).__init__(compress_method)

        self.vision_model = model.visual_encoder
        self.text_model = model.text_encoder 
        self.vision_proj = model.vision_proj 
        self.text_proj = model.text_proj 
        self.compress_layers = [6]

   
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_fourier_signals=False):
        B = pixel_values.shape[0]
        x = self.vision_model.patch_embed(pixel_values)
        hidden_states = []
        energy = [] 
        cls_tokens = self.vision_model.cls_token.expand(
            B, -1, -1
        ) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vision_model.pos_embed[:, : x.size(1), :]
        x = self.vision_model.pos_drop(x)
        for i, blk in enumerate(self.vision_model.blocks):
            if i in self.compress_layers: 
                cls = x[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    x[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state
                )
                x = torch.cat([cls, state], dim=1)

                # print(x.shape)
                if return_all_fourier_signals or i == len(self.vision_model.blocks)-1:
                    energy.append(cur_energy)
                    hidden_states.append(state)

            x = blk(x)
        x = self.vision_model.norm(x)

        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed, hidden_states, energy

    def get_text_features(self, input_ids, attention_mask):
        with torch.no_grad():
            class Text(object):
                pass
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            text_output = self.text_model.forward_text(text)
            last_hidden_state = text_output[0] 
            text_embed = self.text_proj(last_hidden_state[:,0,:])

            return  last_hidden_state, text_embed


class DCTHFClip(CompressedModel):

    def __init__(self, model:AutoModel, compress_method='dct'):
        super(DCTHFClip, self).__init__(compress_method)

        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [12]

    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_fourier_signals=False):
        energy = []
        all_hidden_states = []
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state
                )
                hidden_states = torch.cat([cls, state], dim=1)
                if return_all_fourier_signals or i == len(self.vision_model.encoder.layers)-1:
                    energy.append(cur_energy)
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
        

        return hidden_states, vision_embed, all_hidden_states, energy

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs[1]
        text_embed = self.text_proj(pooled_output)

        return  text_outputs[0], text_embed