
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from .perceiver import FuseMultiModalModel 
from hyptorch.lorentz.layers import LorentzMLR 
from .seq_linear import LorentzSeqLinear
from transformers import PerceiverConfig
from typing import List

class Text(object):
    pass

class BLIPEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:,0,:]
        return last_hidden_state, pooled_output 

class CLIPEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        pooled_output = outputs[1]
        last_hidden_state = outputs[0]
        return last_hidden_state, pooled_output 

class LavisEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body= body 

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is not None:
            outputs = self.body.forward_features(
                pixel_values,
            )
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.body.forward_text(text)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]

        return last_hidden_state, pooled_output


class FuseEncoder(nn.Module): 
    def __init__(self, config, d_visions, d_texts, ft_out, vision_bodies, text_bodies, vision_head, text_head, manifold:CustomLorentz=None ,mapper=None) -> None:
        super().__init__()
        self.manifold = manifold
        self.vision_bodies = nn.ModuleList(vision_bodies) 
        self.text_bodies = nn.ModuleList(text_bodies) 
        self.vision_head = vision_head 
        self.text_head = text_head
        self.config = config
        self.mapper = mapper
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
        self.perceiver_head = FuseMultiModalModel(
            config=head_config, 
            d_visions=d_visions,
            d_texts=d_texts,
            num_blocks=config.num_blocks
        ) 
        self.perceiver_proj_text = LorentzSeqLinear(manifold, ft_in=config.d_latents +1 , layer_dims=[config.d_latents*4+1, ft_out + 1], act_func='gelu', dropout=0.3)
        self.perceiver_proj_vision= LorentzSeqLinear(manifold, ft_in=config.d_latents +1 , layer_dims=[config.d_latents*4+1, ft_out + 1], act_func='gelu', dropout=0.3)
        self.itm_head = LorentzMLR(manifold, config.d_latents + 1, 2)
    

    def forward(
            self,
            pixel_values: List[torch.FloatTensor] = None,
            input_ids: List[torch.Tensor] = None,
            attention_mask: List[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        last_hidden_states = []
        if pixel_values is not None:
            for i in range(len(pixel_values)):
                if i == 0: 
                    last_hidden_state, pooled_output = self.vision_bodies[i](
                        pixel_values=pixel_values[i]
                    )
                    root = self.vision_head(pooled_output)
                else:
                    with torch.no_grad():
                        last_hidden_state, _ = self.vision_bodies[i](
                            pixel_values=pixel_values[i]
                        )
                last_hidden_states.append(last_hidden_state)

            latents_output = self.perceiver_head.get_vision_features(last_hidden_states)
            if self.mapper is not None:
                root = self.mapper(root, use_normalized=True)
                lorentz_latents = self.mapper(latents_output[:,0,:])
            lorentz_latents = self.perceiver_proj_vision(lorentz_latents) 

        else:
            for i in range(len(input_ids)):
                if i == 0:
                    last_hidden_state, pooled_output = self.text_bodies[i](
                        input_ids=input_ids[i], 
                        attention_mask=attention_mask[i]
                    )
                    root = self.text_head(pooled_output)
                else:
                    with torch.no_grad():
                        last_hidden_state, _ = self.text_bodies[i](
                            input_ids=input_ids[i], 
                            attention_mask=attention_mask[i]
                        )
                last_hidden_states.append(last_hidden_state)

            latents_output = self.perceiver_head.get_text_features(last_hidden_states, attention_mask=attention_mask)
            if self.mapper is not None:
                root = self.mapper(root, use_normalized=False)
                lorentz_latents = self.mapper(latents_output[:,0,:])
            lorentz_latents = self.perceiver_proj_text(lorentz_latents)

        if self.manifold is not None:
            output = self.manifold.get_space(lorentz_latents) + self.manifold.get_space(root)
            output = self.manifold.add_time(output)
        else:
            output = lorentz_latents + root

        return latents_output, output    


    def compute_itm(self, vision_hidden_states:torch.Tensor, text_hidden_states:torch.Tensor): 
        itm, _ = self.perceiver_head.compute_itm(
            text_inputs=text_hidden_states, 
            vision_inputs=vision_hidden_states, 
        ) 

        # itm_vision = self.manifold_mapper(itm_vision)
        itm = self.manifold_mapper(itm)
        # hidden_states = torch.cat([itm_vision, itm_text], dim=1)
        itm = self.perceiver_proj(itm) 
        itm_score = self.itm_head(itm)
        return itm_score 







 

