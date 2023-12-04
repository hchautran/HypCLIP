import torch
import torch.nn as nn
from .perceiver_layers import PerceiverLayer
from transformers import PerceiverConfig, PerceiverPreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List

class QLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_text, d_vision, latent_size, num_self_attend=None) -> None:
        super().__init__()
        self.text_cross_attn = PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=d_text,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self.vision_cross_attn = PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=d_vision,
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
                    q_dim=latent_size,
                    kv_dim=latent_size,
                    is_cross_attention=False,
                    use_query_residual=True,
                    num_heads=config.num_self_attention_heads,
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

    def forward(self, text_inputs, vision_inputs, vision_question, text_question,attention_mask=None):
        vision_output = self.vision_cross_attn(
            vision_question,
            inputs=vision_inputs,
        ) 
        text_output = self.text_cross_attn(
            text_question,
            inputs=text_inputs,
            inputs_mask=attention_mask
        ) 
        state = torch.cat([vision_output[0], text_output[0]], dim=1)
        for _, layer_module in enumerate(self.self_attends):
            output = layer_module(state)
            state = output[0] 
        return state
    
    def get_vision_features(self, inputs, question, attention_mask=None):
        cross_output = self.vision_cross_attn(
            question,
            inputs=inputs,
            inputs_mask=attention_mask
        ) 
        state = cross_output[0] 
        for _, layer_module in enumerate(self.self_attends):
            self_output = layer_module(state)
            state = self_output[0] 
        return state, cross_output[0]
    
    
    def get_text_features(self, inputs, question, attention_mask=None):
        cross_output = self.text_cross_attn(
            question,
            inputs=inputs,
            inputs_mask=attention_mask
        ) 
        state = cross_output[0] 
        for _, layer_module in enumerate(self.self_attends):
            self_output = layer_module(state)
            state = self_output[0] 
        return state, cross_output[0]
    
    def compute_itm(self, cross_text_latents, cross_image_latents):
        state = torch.cat([cross_image_latents, cross_text_latents], dim=1)
        for _, layer_module in enumerate(self.self_attends):
            output = layer_module(state)
            state = output[0] 
        return state






class MultiModalModel(nn.Module,  ModuleUtilsMixin):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, num_blocks, mapper=None ) -> None:
        super().__init__()
        self.config = config
        self.num_blocks = num_blocks 
        self.num_latents = config.num_latents
        self.num_cross_heads = config.num_cross_attention_heads
        self.vision_question = nn.Parameter(
            torch.empty(config.num_latents, config.d_latents)
        )
        self.text_question = nn.Parameter(
            torch.empty(12, config.d_latents)
        )
        nn.init.uniform_(self.vision_question.data, -config.initializer_range, config.initializer_range)
        nn.init.uniform_(self.text_question.data, -config.initializer_range, config.initializer_range)
        self.multimodal_layer = QLayer(config=config, d_text=d_text, d_vision=d_vision, latent_size=config.d_latents)

        # self.vision_layer = QLayer(config=config, hidden_size=d_vision, latent_size=config.d_latents ) 
        # self.text_layer = QLayer(config=config, hidden_size=d_text, latent_size=config.d_latents ) 
        # self.itm_head = nn.ModuleList([QLayer(config=config, hidden_size=config.d_latents, latent_size=config.d_latents, num_self_attend=0) for i in range(num_blocks)]) 
        self.mapper = mapper

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def forward(self, vision_inputs:torch.Tensor, text_inputs:torch.Tensor, attention_mask=None):
        bs = vision_inputs.shape[0]
        vision_question = self.vision_question.expand(bs, -1, -1)
        text_question = self.text_question.expand(bs, -1, -1)
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask, input_shape=text_inputs.shape)

        for i in range(self.num_blocks):
            state = self.multimodal_layer(
                text_inputs, 
                vision_inputs, 
                text_question=text_question, 
                vision_question=vision_question, 
                attention_mask=attention_mask
            ) 

        return state 


    def get_vision_features(self, vision_inputs:torch.Tensor):
        bs = vision_inputs.shape[0]
        vision_question = self.vision_question.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_vision, cross_vision = self.multimodal_layer.get_vision_features(vision_inputs, vision_question) 
            vision_question = itc_vision

        return itc_vision, cross_vision

    def get_text_features(self, text_inputs:torch.Tensor, attention_mask:torch.Tensor=None): 
        bs = text_inputs.shape[0]
        text_question = self.text_question.expand(bs, -1, -1)
        attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask, input_shape=text_inputs.shape)

        for i in range(self.num_blocks):
            itc_text, cross_text = self.multimodal_layer.get_text_features(text_inputs, text_question, attention_mask) 
            text_question = itc_text

        return itc_text, cross_text

    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor):
        for i in range(self.num_blocks):
            itm_states = self.multimodal_layer.compute_itm(text_latents, vision_latents) 
        return itm_states 
        


class FuseQLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_texts, d_visions, latent_size, num_self_attend=None) -> None:
        super().__init__()
        self.text_cross_attns = nn.ModuleList([PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=False,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        ) for hidden_size in d_texts])
        self.vision_cross_attns = nn.ModuleList([PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=False,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        ) for hidden_size in d_visions])
        text_self_attends = []
        vision_self_attends = []
        num_self_attends_per_block = num_self_attend if num_self_attend is not None else config.num_self_attends_per_block

        for _ in range(num_self_attends_per_block):
            text_self_attends.append(
                PerceiverLayer(
                    config,
                    q_dim=latent_size,
                    kv_dim=latent_size,
                    is_cross_attention=False,
                    use_query_residual=True,
                    num_heads=config.num_self_attention_heads,
                    widening_factor=config.self_attention_widening_factor,
                )
            )
            vision_self_attends.append(
                PerceiverLayer(
                    config,
                    q_dim=latent_size,
                    kv_dim=latent_size,
                    is_cross_attention=False,
                    use_query_residual=True,
                    num_heads=config.num_self_attention_heads,
                    widening_factor=config.self_attention_widening_factor,
                )
            )
        self.text_self_attends=nn.ModuleList(text_self_attends)
        self.vision_self_attends=nn.ModuleList(vision_self_attends)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, vision_question, text_question, attention_mask=None):
        vision_cross_output = []
        text_cross_output = []
        for i in range(text_inputs):
            vision_output = self.vision_cross_attn[i](
                vision_question,
                inputs=vision_inputs[i],
            ) 
            text_output = self.text_cross_attn[i](
                text_question,
                inputs=text_inputs[i],
                inputs_mask=attention_mask
            ) 
            vision_cross_output.append(vision_output[0])
            text_cross_output.append(text_output[0])
        vision_states = torch.cat([vision_cross_output], dim=1)
        text_states = torch.cat([text_cross_output], dim=1)
        state = torch.cat([vision_states, text_states], dim=1)
        for _, layer_module in enumerate(self.self_attends):
            output = layer_module(state)
            state = output[0] 
        return state
    
    def get_vision_features(self, inputs, question):
        cross_outputs = []
        for i, input in enumerate(inputs):
            cross_output = self.vision_cross_attns[i](
                question,
                inputs=input,
            ) 
            cross_outputs.append(cross_output[0])
        cross_output = torch.cat(cross_outputs, dim=1) 
        state = cross_output
        for _, layer_module in enumerate(self.vision_self_attends):
            self_output = layer_module(state)
            state = self_output[0] 
        return state, cross_output
    
    
    def get_text_features(self, inputs, question, attention_masks):
        cross_outputs = []
        for i, input in enumerate(inputs):
            cross_output = self.text_cross_attns[i](
                question,
                inputs=input,
                attention_mask=attention_masks[i]
            ) 
            cross_outputs.append(cross_output[0])
        cross_output = torch.cat(cross_outputs, dim=1) 
        state = cross_output
        for _, layer_module in enumerate(self.text_self_attends):
            self_output = layer_module(state)
            state = self_output[0] 
        return state, cross_output
    
    
    def compute_itm(self, cross_text_latents, cross_image_latents):
        state = torch.cat([cross_image_latents, cross_text_latents], dim=1)
        for _, layer_module in enumerate(self.vision_self_attends):
            output = layer_module(state)
            state = output[0] 
        return state


class FuseMultiModalModel(PerceiverPreTrainedModel, ModuleUtilsMixin):
    def __init__(self, config:PerceiverConfig, d_visions, d_texts, num_blocks, mapper=None ) -> None:
        super().__init__(config=config)
        self.config = config
        self.num_blocks = num_blocks 
        self.num_latents = config.num_latents
        self.num_cross_heads = config.num_cross_attention_heads
        self.latents = nn.Parameter(
            torch.empty(config.num_latents, config.d_latents)
        )
        self.multimodal_layer = FuseQLayer(config=config, d_texts=d_texts, d_visions=d_visions, latent_size=config.d_latents)
        self.mapper = mapper
        self._init_weights(self)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def forward(self, vision_inputs:torch.Tensor, text_inputs:torch.Tensor, attention_masks=None):
        bs = vision_inputs.shape[0]
        vision_question = self.latents.expand(bs, -1, -1)
        text_question = self.latents.expand(bs, -1, -1)
        if attention_masks is not None:
            attention_masks = [self.get_extended_attention_mask(attention_mask=attention_masks[i], input_shape=text_inputs[i].shape) for i in range(len(attention_masks))]

        for i in range(self.num_blocks):
            state = self.multimodal_layer(
                text_inputs, 
                vision_inputs, 
                text_question=text_question, 
                vision_question=vision_question, 
                attention_mask=attention_masks
            ) 

        return state 


    def get_vision_features(self, vision_inputs:torch.Tensor):
        bs = vision_inputs[0].shape[0]
        vision_question = self.latents.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_vision, cross_vision = self.multimodal_layer.get_vision_features(vision_inputs, vision_question) 
            vision_question = itc_vision

        return itc_vision, cross_vision

    def get_text_features(self, text_inputs:torch.Tensor, attention_masks:torch.Tensor=None): 
        bs = text_inputs[0].shape[0]
        text_question = self.latents.expand(bs, -1, -1)
        attention_masks = [self.get_extended_attention_mask(attention_mask=attention_masks[i], input_shape=text_inputs[i].shape) for i in range(len(attention_masks))]

        for i in range(self.num_blocks):
            itc_text, cross_text = self.multimodal_layer.get_text_features(text_inputs, text_question, attention_masks) 
            text_question = itc_text

        return itc_text, cross_text

    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor):
        for i in range(self.num_blocks):
            itm_states = self.multimodal_layer.compute_itm(text_latents, vision_latents) 
        return itm_states 
        