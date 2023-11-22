import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig

class MultiModalLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, vision_size, text_size, latent_size, num_self_attend=None) -> None:
        super().__init__()
        self.vision_layer= PerceiverLayer(
            config,
            kv_dim=vision_size,
            q_dim=latent_size,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self.text_layer= PerceiverLayer(
            config,
            kv_dim=text_size,
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

    def forward(self, text_inputs, vision_inputs, text_question, vision_question , self_attend_mask=None):
        text_output = self.text_layer(
            text_question,
            inputs=text_inputs,
        )

        vision_output = self.vision_layer(
            vision_question,
            inputs=vision_inputs,
        ) 

        text_state = text_output[0] 
        vision_state = vision_output[0] 
        for layer_module in self.self_attends:
            input_itc = torch.cat([text_state, vision_state], dim=1)
            itc_state = layer_module(input_itc , attention_mask=self_attend_mask)
            text_state, vision_state = torch.split(itc_state[0], [text_state.shape[1], vision_state.shape[1]] ,dim=1)

        return text_state, vision_state

    def get_vision_features(self, vision_inputs, question):
        vision_output = self.vision_layer(
            question,
            inputs=vision_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            vision_state = vision_output[0] 
            vision_output = layer_module(vision_state)
        return vision_output[0]

    def get_text_features(self, text_inputs, question):
        text_output = self.text_layer(
            question,
            inputs=text_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            text_state = text_output[0]
            text_output = layer_module(text_state)
        return text_output[0]
    
 


        

class MultiModalModel(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, num_blocks, mapper=None ) -> None:
        super().__init__()
        self.num_blocks = num_blocks 
        self.num_latents = config.num_latents
        self.vision_question = nn.Parameter(
            torch.zeros(config.num_latents, config.d_latents)
        )
        self.text_question = nn.Parameter(
            torch.zeros(config.num_latents, config.d_latents)
        )
        nn.init.kaiming_uniform_(self.vision_question.data)
        nn.init.kaiming_uniform_(self.text_question.data)
        self.multimodal_layer = MultiModalLayer(config=config, vision_size=d_vision, text_size=d_text ,latent_size=config.d_latents) 
        self.mapper = mapper

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params


    def forward(self, text_inputs:torch.Tensor, vision_inputs:torch.Tensor, self_attend_mask=None, text_ori=None, vision_ori=None):

        bs = text_inputs.shape[0]
        vision_question = self.vision_question.expand(bs, -1, -1) 
        text_question = self.text_question.expand(bs, -1, -1) 

        
        for i in range(self.num_blocks):
            itc_text, itc_vision = self.multimodal_layer(
                text_inputs=text_inputs, 
                vision_inputs=vision_inputs, 
                text_question = text_question,
                vision_question = vision_question,
                self_attend_mask=self_attend_mask
            )
            text_question = itc_text
            vision_question = itc_vision
        
        if self.mapper is not None:
            itc_text = self.mapper(itc_text, use_normalized=False)
            itc_vision = self.mapper(itc_vision, use_normalized=False) 

        return itc_text, itc_vision

    def get_vision_features(self, vision_inputs, vision_ori:torch.Tensor=None):
        bs = vision_inputs.shape[0]
        if vision_ori is not None:
            vision_question = self.vision_question.expand(bs, -1, -1) + vision_ori.expand(self.vision_question.shape[0], -1 ,-1).permute(1,0,2)
        else:
            vision_question = self.vision_question.expand(bs, -1, -1)
        for i in range(self.num_blocks):
            itc_vision = self.multimodal_layer.get_vision_features(vision_inputs, vision_question) 
            vision_question = itc_vision

        if self.mapper is not None:
            itc_vision = self.mapper(itc_vision, use_normalized=False)
        return itc_vision 

    def get_text_features(self, text_inputs, text_ori=None):
        bs = text_inputs.shape[0]
        if text_ori is not None:
            text_question = self.text_question.expand(bs, -1, -1) + text_ori.expand(self.text_question.shape[0], -1 ,-1).permute(1,0,2)
        else:
            text_question = self.text_question.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_text = self.multimodal_layer.get_text_features(text_inputs, text_question) 
            text_question = itc_text

        if self.mapper is not None:
            itc_text = self.mapper(itc_text, use_normalized=False) 
        return itc_text



