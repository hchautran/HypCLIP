import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig
from .seq_linear import SeqLinear
from .graphs import ProjLayers 
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz
from typing import Optional

class MultiModalLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, hidden_size ,num_self_attend=None) -> None:
        super().__init__()
        self.vision_layer= PerceiverLayer(
            config,
            kv_dim=hidden_size,
            q_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self.text_layer= PerceiverLayer(
            config,
            kv_dim=hidden_size,
            q_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self_attention_layers = []
        num_self_attends_per_block = num_self_attend if num_self_attend is not None else config.num_self_attends_per_block
        for _ in range(num_self_attends_per_block):
            self_attention_layers.append(
                PerceiverLayer(
                    config,
                    is_cross_attention=False,
                    num_heads=config.num_self_attention_heads,
                    q_dim=hidden_size,
                    kv_dim=hidden_size,
                    use_query_residual=True,
                    widening_factor=config.self_attention_widening_factor,
                )
            )
        self.self_attends=nn.ModuleList(self_attention_layers)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, text_question, vision_question , self_attend_mask=None):
        text_output = self.text_layer(
            text_question,
            inputs=text_inputs,
        )

        vision_output = self.vision_layer(
            vision_question,
            inputs=vision_inputs,
        ) 
        text_itc_state = text_output[0] 
        vision_itc_state = vision_output[0] 
        text_itm_state = text_output[0] 
        vision_itm_state = vision_output[0] 

        for layer_module in self.self_attends:
            input_itc = torch.cat([text_itc_state, vision_itc_state], dim=1)
            input_itm = torch.cat([text_itm_state, vision_itm_state], dim=1)
            itc_state = layer_module(input_itc , attention_mask=self_attend_mask)
            itm_state = layer_module(input_itm, attention_mask=None)

            text_itc_state, vision_itc_state = torch.split(itc_state[0], [text_itc_state.shape[1], vision_itc_state.shape[1]] ,dim=1)
            text_itm_state, vision_itm_state = torch.split(itm_state[0], [text_itc_state.shape[1], vision_itc_state.shape[1]] ,dim=1)

        return text_itc_state, vision_itc_state, text_itm_state, vision_itm_state

    def get_vision_features(self, vision_inputs, question):
        vision_output = self.vision_layer(
            question,
            inputs=vision_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            vision_state = vision_output[0] 
            vision_output = layer_module(vision_state)
        return vision_output[0]

    def get_text_features(self, text_inputs, question, attention_mask=None):
        text_output = self.text_layer(
            question,
            inputs=text_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            text_state = text_output[0]
            text_output = layer_module(text_state)
        return text_output[0]


        

class MultiModalModel(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, d_out, num_vision_blocks, num_text_blocks, mapper=None ) -> None:
        super().__init__()
        self.num_text_blocks = num_text_blocks
        self.num_vision_blocks = num_vision_blocks
        text_sizes = [d_text] * num_text_blocks
        vision_sizes = [d_vision] * num_vision_blocks 
        self.num_latents = config.num_latents
        
        self.vision_proj_layers = ProjLayers(vision_sizes, hidden_sizes=[512,512, d_out], dropout=0.4)
        self.text_proj_layers = ProjLayers(text_sizes, hidden_sizes=[512,512, d_out], dropout=0.4)


        self.layers = MultiModalLayer(config=config, hidden_size=d_out) 
        self.mapper = mapper

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def get_proj_output(self, hidden_states, proj_layers, num_layers):
        # bs = pooled_output.shape[0]
        hidden_states = hidden_states[(len(hidden_states) - num_layers):] 
        output = torch.cat(proj_layers(hidden_states), dim =-2)
        # output = torch.cat([pooled_output.expand(bs,-1,-1),output], dim =-2)
        return output



    def forward(self, text_ori:torch.Tensor, vision_ori:torch.Tensor ,text_inputs:torch.Tensor, vision_inputs:torch.Tensor, self_attend_mask=None):

        text_inputs = self.get_proj_output(text_inputs, proj_layers=self.text_proj_layers, num_layers=self.num_text_blocks)
        vision_inputs = self.get_proj_output(vision_inputs, proj_layers=self.vision_proj_layers, num_layers=self.num_vision_blocks)
        # print(text_ori.shape)
        # print(self.text_question.shape)

        # print(text_question.shape)
        # print(vision_question.shape)

        text_question = text_ori.expand([self.num_latents, -1, -1]).transpose(0,1) 
        vision_question = vision_ori.expand([self.num_latents, -1, -1]).transpose(0, 1)


        itc_text, itc_vision, itm_text_state, itm_vision_state = self.layers(
            text_inputs=text_inputs, 
            vision_inputs=vision_inputs, 
            text_question = text_question,
            vision_question = vision_question,
            self_attend_mask=self_attend_mask
        )

        itm_text = torch.mean(itm_text_state, dim=-2)
        itm_vision = torch.mean(itm_vision_state, dim=-2) 

        itc_text= itc_text[:,0,:]
        itc_vision= itc_vision[:,0,:]

        if self.mapper is not None:
            itc_text = self.mapper(itc_text)
            itc_vision = self.mapper(itc_vision, use_normalized=True) 
            itm_text = self.mapper(itm_text)
            itm_vision = self.mapper(itm_vision, use_normalized=True)

        return itc_text, itc_vision, itm_text, itm_vision

    def get_vision_features(self, vision_ori, vision_inputs):
        bs = vision_ori.size(0)
        vision_inputs = self.get_proj_output(vision_inputs, proj_layers=self.vision_proj_layers, num_layers=self.num_vision_blocks)
        vision_question = vision_ori.expand([self.num_latents, -1, -1]).transpose(0, 1)

        itc_vision = self.layers.get_vision_features(vision_inputs, vision_question) 
        itc_vision = itc_vision[:,0,:]

        if self.mapper is not None:
            itc_vision = self.mapper(itc_vision, use_normalized=True)
        return itc_vision 

    def get_text_features(self, text_ori, text_inputs):
        bs = text_ori.size(0)
        text_inputs = self.get_proj_output(text_inputs, proj_layers=self.text_proj_layers, num_layers=self.num_text_blocks)
        text_question = text_ori.expand([self.num_latents, -1, -1]).transpose(0,1) 

        itc_text = self.layers.get_text_features(text_inputs, text_question)
        itc_text= itc_text[:,0,:]

        if self.mapper is not None:
            itc_text = self.mapper(itc_text) 
        return itc_text



