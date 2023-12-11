import torch
import torch.nn as nn
from .perceiver_layers import PerceiverLayer
from .perceiver_lorentz_layers import PerceiverLayer as LorentzPerceiverLayers 
from transformers import PerceiverConfig, PerceiverPreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List
class FuseQLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_texts, d_visions, latent_size, num_self_attend=None) -> None:
        super().__init__()
        self.text_cross_attns = nn.ModuleList([PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        ) for hidden_size in d_texts])
        self.vision_cross_attns = nn.ModuleList([PerceiverLayer(
            config,
            q_dim=latent_size,
            kv_dim=hidden_size,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        ) for hidden_size in d_visions])
        self_attends = []
        num_self_attends_per_block = num_self_attend if num_self_attend is not None else config.num_self_attends_per_block

        for _ in range(num_self_attends_per_block):
            self_attends.append(
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
         
        self.self_attends=nn.ModuleList(self_attends)

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
        # for _, layer_module in enumerate(self.self_attends):
        #     output = layer_module(state)
        #     state = output[0] 
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
        for _, layer_module in enumerate(self.self_attends):
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

        return state, cross_output
    
    
    def compute_itm(self, cross_text_latents, cross_image_latents):
        itm_text= torch.cat([cross_text_latents, cross_image_latents], dim=1)
        for i in range(len(self.self_attends)):
            itm_state = self.self_attends[i](itm_text)[0]
        return itm_state 


class FuseMultiModalModel(PerceiverPreTrainedModel, ModuleUtilsMixin):
    def __init__(self, config:PerceiverConfig, d_visions, d_texts, num_blocks, mapper=None ) -> None:
        super().__init__(config=config)
        self.config = config
        self.num_blocks = num_blocks 
        self.num_latents = config.num_latents
        self.num_cross_heads = config.num_cross_attention_heads
        self.vision_latent_projs = nn.ModuleList([])
        self.text_latent_projs = nn.ModuleList([])
        self.latents = nn.Parameter(torch.empty(config.num_latents, config.d_latents))

      
                
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
    


    def get_vision_features(self, vision_inputs:torch.Tensor, vision_oris:torch.Tensor):
        bs = vision_inputs[0].shape[0]

        vision_questions = vision_oris.unsqueeze(1) + self.latents.expand(bs, -1, -1)

        for i in range(self.num_blocks):
            itc_vision, cross_vision = self.multimodal_layer.get_vision_features(vision_inputs, vision_questions) 
            vision_questions = itc_vision

        return itc_vision, cross_vision

    def get_text_features(self, text_inputs:torch.Tensor, text_oris:torch.Tensor, attention_masks:torch.Tensor=None): 
        bs = text_inputs[0].shape[0]

        text_questions = text_oris.unsqueeze(1) + self.latents.expand(bs, -1, -1)
        # print(text_questions.shape)
        attention_masks = [self.get_extended_attention_mask(attention_mask=attention_masks[i], input_shape=text_inputs[i].shape) for i in range(len(attention_masks))]

        for i in range(self.num_blocks):
            itc_text, cross_text = self.multimodal_layer.get_text_features(text_inputs, text_questions, attention_masks) 
            text_questions = itc_text

        return itc_text, cross_text

    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor):
        for i in range(self.num_blocks):
            itm_states = self.multimodal_layer.compute_itm(text_latents, vision_latents) 
        return itm_states 
        