
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
)
import math
from lavis import BlipRetrieval, Blip2Qformer
from transformers import AutoModel 
from .dct import dct, idct


class CompressedModel(nn.Module):
    def __init__(self, compress_method='dct', r=0.9, window_size=12):
        super().__init__()
        self.r = r
        self.window_size=window_size
        self.compress_method = compress_method
    

    def std_filter(self, x, percentile_threshold, filter_strategy='std'):
        percentile_threshold = percentile_threshold
        std_array = x.mean(0).std(1)
        threshold = torch.quantile(std_array, percentile_threshold, dim=-1, keepdim=True)
        x_filtered = []
        for i in range(0,std_array.shape[0], self.window_size):
            if i + self.window_size <= std_array.shape[0]:
                cur_window = x[:, i:i+self.window_size, :].clone()
                cur_std_array = std_array[i: i+ self.window_size].clone()
                if cur_std_array.max() > threshold:
                    x_filtered.append(cur_window)
                elif filter_strategy == 'std':
                    x_filtered.append((cur_window.permute(0, 2, 1) @ cur_std_array.expand(x.shape[0], -1).unsqueeze(2)).permute(0,2,1))
                elif filter_strategy == 'mean':
                    x_filtered.append(torch.mean(cur_window, dim=1, keepdim=True))
            else:
                x_filtered.append(x[:,i:,:])
        return torch.cat(x_filtered, dim=1)

    def std_filter_with_r(self, x):        
        B, T, D = x.shape
        k = math.floor((T- T*self.r)/self.window_size)
        first_x = x[:,:(T%self.window_size),:]
        remain_x = x[:,(T%self.window_size):,:]
        remain_x = remain_x.view(B, int(remain_x.shape[1]/self.window_size), - 1, D)
        std_array = remain_x.std(-1)
        max_std, _ = std_array.mean(0).max(-1) 
       
        min_indices = torch.sort(torch.topk(max_std, k=k, dim=-1, largest=False)[1]).values
        output = [first_x]
        prev_i = -1
        
        for i in min_indices:
            output.append(remain_x[:, prev_i + 1:i, :, :]. view(B, -1, D))
            prev_i = i
            min_window = remain_x[:, i, :, :]
            # min_std_array = std_array[:, i, :]
            # if filter_strategy == 'std':
            #     min_std_array = min_std_array - min_std_array.min()
            #     min_std_array = min_std_array/min_std_array.max()
            #     min_window = (min_window.permute(0, 2, 1) @ min_std_array.unsqueeze(2)).permute(0,2,1)
            # else:
            min_window = torch.mean(min_window, dim=1, keepdim=True)

            output.append(min_window)
            torch.flops

        output.append(remain_x[:, prev_i+1:, :, :].view(B, -1, D))
        return torch.cat(output, dim=1), None

    def random_filter_with_r(self, x):        
        B, T, D = x.shape
        k = math.floor((T- T*self.r)/self.window_size)
        first_x = x[:,:(T%self.window_size),:]
        remain_x = x[:,(T%self.window_size):,:]
        remain_x = remain_x.view(B, int(remain_x.shape[1]/self.window_size), - 1, D)
        indices = torch.sort(torch.randint(0, remain_x.shape[1], k))
        output = [first_x]
        prev_i = -1
        
        for i in indices:
            output.append(remain_x[:, prev_i + 1:i, :, :]. view(B, -1, D))
            prev_i = i
            cur_window = remain_x[:, i, :, :]
            cur_window = torch.mean(cur_window, dim=1, keepdim=True)
            output.append(cur_window)

        output.append(remain_x[:, prev_i+1:, :, :].view(B, -1, D))
        return torch.cat(output, dim=1), None
    
    
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

    def dc_transform(self, x, use_reconstucted_state=False, threshold=None):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        k = math.ceil(self.r * T)

        if use_reconstucted_state:
            x_dct = x_dct[:k, :, :]
            x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2)
            # print(x)
   
        return x.permute(1,0,2), x_dct.permute(1,0,2)


    def direct(self, x, use_reconstucted_state = False):
        k = math.ceil(0.90 * x.shape[1])
        if use_reconstucted_state:
            x = x[:,:k,:]  
        return x, x
    
    def std_based_compress(self, x, use_reconstucted_state, threshold=0.7,filter_strategy='std'):
        if use_reconstucted_state:
            x = self.std_filter(x, threshold, filter_strategy=filter_strategy) 
        return x, x
   

    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        raise NotImplementedError("This method is not implemented yet")

    def get_text_features(self, input_ids, attention_mask):
        raise NotImplementedError("This method is not implemented yet")
    
    def compress_hidden_state(self, x, use_compressed_hidden_state, threshold=0.7):
        if self.compress_method == 'dct':
            x_reconstructed, energy = self.dc_transform(x ,use_compressed_hidden_state ) 
        elif self.compress_method == 'std':
            # x_reconstructed, energy = self.std_based_compress(x ,use_compressed_hidden_state, threshold=threshold, window_size=window_size, filter_strategy='std') 
            x_reconstructed, energy = self.random_filter_with_r(x , filter_strategy='random') 
        elif self.compress_method == 'mean':
            # x_reconstructed, energy = self.std_based_compress(x ,use_compressed_hidden_state, threshold=threshold, window_size=window_size, filter_strategy='mean') 
            x_reconstructed, energy = self.std_filter_with_r(x , filter_strategy='std', threshold=threshold) 
        elif self.compress_method == 'direct':
            x_reconstructed, energy = self.direct(x ,use_compressed_hidden_state) 
        else: 
            return x, x

        return  x_reconstructed, energy

    
class CompressedHFBLIP(CompressedModel):
    config_class = BlipConfig

    def __init__(self, model:AutoModel, compress_method='dct'):
        super(CompressedHFBLIP, self).__init__(compress_method)
        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [6, 7, 8]
     

    
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        hidden_states = self.vision_model.embeddings(pixel_values)
        all_hidden_states = []
        energy = []
        real_mem = 0
        total_mem = 0
        ori_size = hidden_states.shape[1]

        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:    
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                    r=0.9
                )
                hidden_states = torch.cat([cls, state], dim=1)
                if return_all_hidden_state or i == len(self.vision_model.encoder.layers)-1:
                    energy.append(cur_energy)
                    all_hidden_states.append(hidden_states)
                real_mem += hidden_states.shape[1]
                total_mem += ori_size 

            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]


        last_hidden_state = self.vision_model.post_layernorm(hidden_states)
        pooled_output = last_hidden_state[:, 0, :]
        vision_embed = self.vision_proj(pooled_output)
       
        return hidden_states, vision_embed, all_hidden_states, energy, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = text_output[0] 
        text_embed = self.text_proj(text_output[1])

        return  last_hidden_state, text_embed


class CompressedLAVISBLIP(CompressedModel):

    def __init__(self, model:BlipRetrieval, compress_method='dct'):
        super(CompressedLAVISBLIP, self).__init__(compress_method)

        self.vision_model = model.visual_encoder
        self.text_model = model.text_encoder 
        self.vision_proj = model.vision_proj 
        self.text_proj = model.text_proj 
        self.compress_layers = [i for i in range(1,len(self.vision_model.blocks))]

   
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
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
        ori_size = x.shape[1]
        real_mem = 0
        total_mem = 0
        for i, blk in enumerate(self.vision_model.blocks):
            if i in self.compress_layers: 
                cls = x[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    x[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                )
                x = torch.cat([cls, state], dim=1)

                if return_all_hidden_state or i == len(self.vision_model.blocks)-1:
                    energy.append(cur_energy)
                    hidden_states.append(state)
                real_mem += x.shape[1]
                total_mem += ori_size 
            x = blk(x)

        # with torch.no_grad():
        x = self.vision_model.norm(x)
        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed, hidden_states, energy, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        # with torch.no_grad():
        class Text(object):
            pass
        text = Text() 
        text.input_ids=input_ids
        text.attention_mask=attention_mask
        text_output = self.text_model.forward_text(text)
        last_hidden_state = text_output[0] 
        text_embed = self.text_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed


class CompressedHFCLIP(CompressedModel):

    def __init__(self, model:AutoModel, compress_method='dct'):
        super(CompressedHFCLIP, self).__init__(compress_method)

        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [15, 16, 18 ,19, 20] if len(self.vision_model.encoder.layers) > 12 else [6, 7, 8]

    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        energy = []
        all_hidden_states = []
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        real_mem = 0
        total_mem = 0
        ori_size = hidden_states.shape[1]
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                    r=self.r
                )
                hidden_states = torch.cat([cls, state], dim=1)
                # print(hidden_states.shape)
                if return_all_hidden_state or i == len(self.vision_model.encoder.layers)-1:
                    energy.append(cur_energy)
                    all_hidden_states.append(hidden_states)
                real_mem += hidden_states.shape[1]
                total_mem += ori_size 

            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        vision_embed = self.vision_proj(pooled_output)
        

        return hidden_states, vision_embed, all_hidden_states, energy, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs[1]
        text_embed = self.text_proj(pooled_output)

        return  text_outputs[0], text_embed

        
class CompressedLAVISBLIP2(CompressedModel):

    def __init__(self, model:Blip2Qformer, compress_method='dct'):
        super(CompressedLAVISBLIP2, self).__init__(compress_method)

        self.ln_vision = model.ln_vision
        self.visual_encoder = model.visual_encoder
        self.query_tokens = model.query_tokens
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.Qformer = model.Qformer
        self.itm_head = model.itm_head
        # self.compress_layers = [20,22,24,26,28,30,32,34,36,38,40]
        
        # self.compress_layers = [24,26,28,30,32,34,36,38,40,42,44,46]
        self.compress_layers = [i for i in range(1,len(self.visual_encoder.blocks))]

   
    def get_vision_features(self, pixel_values:torch.Tensor, use_compressed_hidden_state=True, return_all_hidden_state=False):
        all_hidden_states = []
        energy = []
        total_mem=0
        real_mem=0
        with torch.no_grad():
            x = self.visual_encoder.patch_embed(pixel_values)
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.visual_encoder.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if self.visual_encoder.pos_embed is not None:
                x = x + self.visual_encoder.pos_embed
            x = self.visual_encoder.pos_drop(x)
            ori_size = x.shape[1]

            rel_pos_bias = self.visual_encoder.rel_pos_bias() if self.visual_encoder.rel_pos_bias is not None else None
            for i, blk in enumerate(self.visual_encoder.blocks):
                if i in self.compress_layers:
                    x, cur_energy = self.compress_hidden_state(
                        x, 
                        use_compressed_hidden_state=use_compressed_hidden_state,
                    )
                x = blk(x, rel_pos_bias)
                if return_all_hidden_state or i == len(self.visual_encoder.blocks) - 1:
                    energy.append(cur_energy)
                    all_hidden_states.append(x)
                real_mem += x.shape[1]
                total_mem += ori_size 
            # vit_embeds = self.visual_encoder(pixel_values)
            vit_embeds = self.ln_vision(x)



        image_atts = torch.ones(vit_embeds.size()[:-1], dtype=torch.long).to(
            pixel_values.device
        )
        query_tokens = self.query_tokens.expand(vit_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vit_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        pooled_output = self.vision_proj(query_output.last_hidden_state)
        # return vit_embeds, pooled_output, all_hidden_states
        return vit_embeds, pooled_output, all_hidden_states, energy, real_mem/total_mem 

    def get_text_features(self, input_ids, attention_mask):
        # with torch.no_grad():
        text_output = self.Qformer.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled_output = self.text_proj(text_output.last_hidden_state[:, 0, :])
        return text_output.last_hidden_state, pooled_output