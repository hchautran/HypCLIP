
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt import PoincareBall 
from .perceiver import FuseMultiModalModel 
from hyptorch.lorentz.layers import LorentzMLR 
from hyptorch.poincare.layers import UnidirectionalPoincareMLR 
from .seq_linear import LorentzSeqLinear, SeqLinear, HypSeqLinear
from transformers import PerceiverConfig
from typing import List, Union

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
    def __init__(self, config, d_visions, d_texts, ft_out, vision_bodies, text_bodies, vision_head, text_head, manifold:Union[CustomLorentz, PoincareBall]=None ,mapper=None) -> None:
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
            self_attention_widening_factor=2,
            cross_attention_widening_factor=2,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        self.perceiver_head = FuseMultiModalModel(
            config=head_config, 
            d_visions=d_visions,
            d_texts=d_texts,
            num_blocks=config.num_blocks
        ) 
        if isinstance(self.manifold, CustomLorentz):
            self.vision_perceiver_proj= LorentzSeqLinear(manifold, ft_in=config.d_latents +1 , layer_dims=[config.d_latents*4+1, ft_out + 1], act_func='gelu', dropout=0.3)
            self.text_perceiver_proj= LorentzSeqLinear(manifold, ft_in=config.d_latents +1 , layer_dims=[config.d_latents*4+1, ft_out + 1], act_func='gelu', dropout=0.3)
            self.itm_head = LorentzMLR(manifold, config.d_latents + 1, 2)
        elif isinstance(self.manifold, PoincareBall):
            self.vision_perceiver_proj= HypSeqLinear(manifold, ft_in=config.d_latents, layer_dims=[config.d_latents*4, ft_out], act_func='gelu', dropout=0.3)
            self.text_perceiver_proj= HypSeqLinear(manifold, ft_in=config.d_latents, layer_dims=[config.d_latents*4, ft_out], act_func='gelu', dropout=0.3)
            self.itm_head = UnidirectionalPoincareMLR(ball=manifold, feat_dim=config.d_latents, num_outcome=2)
        else: 
            self.vision_perceiver_proj= SeqLinear(ft_in=config.d_latents, layer_dims=[config.d_latents*4, config.d_latents*4,ft_out], act_func='gelu', dropout=0.3)
            self.text_perceiver_proj= SeqLinear(ft_in=config.d_latents, layer_dims=[config.d_latents*4, config.d_latents*4,ft_out], act_func='gelu', dropout=0.3)
            self.itm_head = nn.Linear(config.d_latents, 2)
    

    def forward(
            self,
            pixel_values: List[torch.FloatTensor] = None,
            input_ids: List[torch.Tensor] = None,
            attention_masks: List[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        last_hidden_states = []
        if pixel_values is not None:
            for i in range(len(pixel_values)):
                # print(pixel_values[i].shape)
                if i == 0: 
                    # with torch.no_grad():
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

            latents_output, cross_latents = self.perceiver_head.get_vision_features(last_hidden_states)
            if self.mapper is not None:
                root = self.mapper(root, use_normalized=True)
                latents_output = self.mapper(latents_output[:,0,:])
            else:
                latents_output = latents_output[:,0,:]
            latents_output = self.vision_perceiver_proj(latents_output)
        else:
            for i in range(len(input_ids)):
                if i == 0:
                    # with torch.no_grad():
                    last_hidden_state, pooled_output = self.text_bodies[i](
                        input_ids=input_ids[i], 
                        attention_mask=attention_masks[i]
                    )
                    root = self.text_head(pooled_output)
                else:
                    with torch.no_grad():
                        last_hidden_state, _ = self.text_bodies[i](
                            input_ids=input_ids[i], 
                            attention_mask=attention_masks[i]
                        )
                last_hidden_states.append(last_hidden_state)

            latents_output, cross_latents = self.perceiver_head.get_text_features(last_hidden_states, attention_masks=attention_masks)
            if self.mapper is not None:
                root = self.mapper(root, use_normalized=False)
                latents_output = self.mapper(latents_output[:,0,:])
            else:
                latents_output = latents_output[:,0,:]
            latents_output = self.text_perceiver_proj(latents_output)

        if isinstance(self.manifold, CustomLorentz):
            # output = self.manifold.pt_addition(root, latents_output) 
            output = self.manifold.get_space(latents_output) + self.manifold.get_space(root)
            output = self.manifold.add_time(output)
            # output = latents_output 
        elif isinstance(self.manifold, PoincareBall):
            output = self.manifold.mobius_add(root, latents_output)
            # output = self.manifold.get_space(latents_output) + self.manifold.get_space(root)
            # output = self.manifold.add_time(output)
        else:
            output = latents_output + output

        return cross_latents, output    


    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor): 
        itm = self.perceiver_head.compute_itm(
            vision_latents=vision_latents, 
            text_latents=text_latents, 
        ) 

        # itm_vision = self.manifold_mapper(itm_vision)
        if self.mapper is not None:
            itm = self.mapper(itm)
        # hidden_states = torch.cat([itm_vision, itm_text], dim=1)
        # itm = self.perceiver_proj(itm) 
        itm_score = self.itm_head(itm).mean(dim=1)
        return itm_score 







 

