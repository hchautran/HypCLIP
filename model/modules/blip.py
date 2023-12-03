
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from .perceiver import MultiModalModel 
from hyptorch.lorentz.layers import LorentzLinear
from .seq_linear import  SeqLinear
from transformers import PerceiverConfig
class Text(object):
    pass

class LavisEncoder(nn.Module): 
    def __init__(self, config, vision_body, vision_head, text_body, text_head, mapper=None, use_normalized=False) -> None:
        super().__init__()
        self.vision_body = vision_body
        self.vision_head = vision_head 
        self.text_body = text_body
        self.text_head = text_head 
        self.config = config
        self.mapper = mapper
        self.use_normalized = use_normalized

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is not None:
            # with torch.no_grad():
            outputs = self.vision_body.forward_features(
                pixel_values,
            )
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.vision_head(pooled_output)
            if self.mapper is not None:
                    pooled_output = self.mapper(pooled_output, use_normalized=True)
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.text_body.forward_text(text)

            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.text_head(pooled_output)
            if self.mapper is not None:
                pooled_output = self.mapper(pooled_output, use_normalized=True)


        return last_hidden_state, pooled_output


class LavisBLIPGraphModel(nn.Module): 
    def __init__(self, d_vision, d_text ,ft_out, config , text_body, text_head, vision_body, vision_head ) -> None:
        super().__init__()
        self.config = config
        self.text_body = text_body
        self.text_head = text_head 
        self.vision_body = vision_body
        self.vision_head = vision_head 
        head_config = PerceiverConfig(
            d_latents=config.d_latents, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            self_attention_widening_factor=2,
            cross_attention_widening_factor=2,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        self.perceiver_head = MultiModalModel(
            config=head_config, 
            d_vision=d_vision,
            d_text=d_text,
            num_blocks=2
        ) 
        # self.perceiver_proj = LorentzSeqLinear(manifold, ft_in=config.d_latents +1 , layer_dims=[512, 512, 512, config.d_latents + 1], act_func='gelu')
        self.perceiver_proj = SeqLinear(ft_in=config.d_latents, layer_dims=[512, 512, 512, config.d_latents], act_func='gelu')
        # self.perceiver_proj = nn.Linear(config.d_latents, ft_out) 
        # self.itm_head = LorentzMLR(manifold, config.d_latents + 1, 2)
        self.itm_head = nn.Linear(config.d_latents , 2)
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        if pixel_values is not None:
            outputs = self.vision_body.forward_features(pixel_values)
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]

            vision_inputs = self.perceiver_head.get_vision_features(last_hidden_state)
            pooled_output = self.vision_head(pooled_output)
            # vision_inputs = self.perceiver_proj(vision_inputs) 

            hidden_states = self.perceiver_proj(vision_inputs) 
            
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.text_body.forward_text(text)

            last_hidden_state = outputs.last_hidden_state[:,0,:].unsqueeze(1)

            text_inputs = self.perceiver_head.get_text_features(last_hidden_state)
            pooled_output = self.text_head(outputs.last_hidden_state[:,0,:])
            # text_inputs = self.perceiver_proj(text_inputs) 
            hidden_states = self.perceiver_proj(text_inputs) 

        output = torch.mean(hidden_states, dim=1)
        output = output + pooled_output
        return last_hidden_state, output
    
    def compute_itm(self, vision_hidden_states:torch.Tensor, text_hidden_states:torch.Tensor):

        itm_text, itm_vision = self.perceiver_head(
            text_inputs=text_hidden_states, 
            vision_inputs=vision_hidden_states, 
            self_attend_mask=None,
        ) 

        hidden_states = torch.cat([itm_vision, itm_text], dim=1)
        hidden_states = self.perceiver_proj(hidden_states) 
        with torch.no_grad():
            itm_score = self.itm_head(hidden_states)
        return itm_score 
        
    
class LavisLorentzBLIPGraphModel(nn.Module): 
    def __init__(self, manifold:CustomLorentz, d_vision, d_text ,ft_out, config, text_body, text_head, vision_body, vision_head,  manifold_mapper, itm_head=None) -> None:
        super().__init__()
        self.config = config
        self.text_body = text_body
        self.text_head = text_head 
        self.vision_body = vision_body
        self.vision_head = vision_head 
        self.manifold = manifold
        self.manifold_mapper = manifold_mapper
        head_config = PerceiverConfig(
            d_latents=config.d_latents, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            self_attention_widening_factor=4,
            cross_attention_widening_factor=4,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        self.perceiver_head = MultiModalModel(
            config=head_config, 
            d_vision=d_vision,
            d_text=d_text,
            num_blocks=config.num_blocks
        ) 
        self.perceiver_hidden_layers= SeqLinear(ft_in=config.d_latents , layer_dims=[config.d_latents*4, config.d_latents*4 , config.d_latents], act_func='gelu', dropout=0.2)
        self.perceiver_proj_text = LorentzLinear(manifold, config.d_latents +1 , ft_out + 1, dropout=0.1)
        self.perceiver_proj_vision = LorentzLinear(manifold, config.d_latents +1 , ft_out + 1, dropout=0.1)
        self.itm_head = nn.Linear(config.d_latents, 2) 
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        if pixel_values is not None:
            outputs = self.vision_body.forward_features(pixel_values)
            last_hidden_state = outputs
            pooled_output = self.vision_head(last_hidden_state[:, 0, :])
            vision_state, latents_output = self.perceiver_head.get_vision_features(last_hidden_state)

            vision_state = self.perceiver_hidden_layers(vision_state[:, 0, :])
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=False)
            vision_state = self.manifold_mapper(vision_state)
            lorentz_latents = self.perceiver_proj_vision(vision_state) 
            
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.text_body.forward_text(text)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = self.text_head(outputs.last_hidden_state[:,0,:])
            text_state, latents_output = self.perceiver_head.get_text_features(last_hidden_state, attention_mask=attention_mask)

            text_state = self.perceiver_hidden_layers(text_state[:,0,:])
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
            text_state = self.manifold_mapper(text_state)
            lorentz_latents = self.perceiver_proj_text(text_state) 

        output = self.manifold.get_space(lorentz_latents) + self.manifold.get_space(pooled_output)
        output = self.manifold.add_time(output)
        return latents_output, output, pooled_output
    
    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor):

        itm_output = self.perceiver_head.compute_itm(
            vision_latents=vision_latents, 
            text_latents=text_latents, 
        ) 

        itm_output = self.perceiver_hidden_layers(itm_output)
        itm_score = self.itm_head(itm_output).mean(dim=1)
        return itm_score
        