import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig
from typing import List

class QLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, hidden_size, latent_size, num_self_attend=None) -> None:
        super().__init__()
        self.cross_attn = PerceiverLayer(
            config,
            kv_dim=hidden_size,
            q_dim=latent_size,
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
                    q_dim=latent_size,
                    kv_dim=latent_size,
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

    def forward(self, inputs, question):
        output = self.cross_attn(
            question,
            inputs=inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            state = output[0] 
            output = layer_module(state)
        return output[0]


class MultiModalModel(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, num_blocks, mapper=None ) -> None:
        super().__init__()
        self.num_blocks = num_blocks 
        self.num_latents = config.num_latents
        self.vision_question = nn.Parameter(
            torch.empty(config.num_latents, config.d_latents)
        )
        self.text_question = nn.Parameter(
            torch.empty(12, config.d_latents)
        )
        nn.init.uniform_(self.vision_question.data, -config.initializer_range, config.initializer_range)
        nn.init.uniform_(self.text_question.data, -config.initializer_range, config.initializer_range)

        self.vision_layer = QLayer(config=config, hidden_size=d_vision, latent_size=config.d_latents) 
        self.text_layer = QLayer(config=config, hidden_size=d_text, latent_size=config.d_latents) 
        self.itm_head = QLayer(config=config, hidden_size=config.d_latents, latent_size=config.d_latents) 
        self.mapper = mapper

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def forward(self, vision_inputs:torch.Tensor, text_inputs:torch.Tensor):
        bs = vision_inputs.shape[0]
        vision_question = self.vision_question.expand(bs, -1, -1)
        text_question = self.text_question.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_text = self.text_layer(text_inputs, text_question) 
            itc_vision = self.vision_layer(vision_inputs, vision_question) 
            vision_question = itc_vision
            text_question = itc_text

        return itc_text, itc_vision


    def get_vision_features(self, vision_inputs:torch.Tensor):
        bs = vision_inputs.shape[0]
        vision_question = self.vision_question.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_vision = self.vision_layer(vision_inputs, vision_question) 
            vision_question = itc_vision

        return itc_vision 

    def get_text_features(self, text_inputs:torch.Tensor): 
        bs = text_inputs.shape[0]
        text_question = self.text_question.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_text = self.text_layer(text_inputs, text_question) 
            text_question = itc_text

        return itc_text

    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor):
        output = self.itm_head(vision_latents, text_latents) 
        return output 
        

