
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt import PoincareBall 
from .perceiver import FuseMultiModalModel 
from hyptorch.lorentz.layers import LorentzMLR, LorentzLinear 
from hyptorch.poincare.layers import UnidirectionalPoincareMLR 
from .seq_linear import LorentzSeqLinear, SeqLinear, HypSeqLinear
from transformers import PerceiverConfig
from typing import List, Union

class Text(object):
    pass

class BLIPEncoder(nn.Module): 
    def __init__(self, body, head) -> None:
        super().__init__()
        self.body = body
        self.head = head
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        num_hidden_states: Optional[int] = None,
        use_first_layers: Optional[bool] = True,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=True
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True
            )

        if not use_first_layers:
            hidden_state = torch.cat(
                outputs.hidden_states[(len(outputs.hidden_states) - num_hidden_states):], dim=1
            )
        else:
            hidden_state = torch.cat(
                outputs.hidden_states[:num_hidden_states], dim=1
            )
        pooled_output = outputs[0][:,0,:]
        final_feat = self.head(pooled_output) 
        return hidden_state, pooled_output, final_feat 

class CLIPEncoder(nn.Module): 
    def __init__(self, body, head) -> None:
        super().__init__()
        self.body = body
        self.head = head
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        num_hidden_states: Optional[int] = None,
        use_first_layers: Optional[bool] = True,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=True
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True
            )

        pooled_output = outputs[1]
        final_feat = self.head(pooled_output)
        if not use_first_layers:
            hidden_state = torch.cat(
                outputs.hidden_states[(len(outputs.hidden_states) - num_hidden_states):], 
                dim=1
            )
        else:
            hidden_state = torch.cat(
                outputs.hidden_states[:num_hidden_states], 
                dim=1
            )
        return hidden_state, pooled_output, final_feat 

class LavisEncoder(nn.Module): 
    def __init__(self, body, head) -> None:
        super().__init__()
        self.body = body 
        self.head = head

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
        
        final_feat = self.head(pooled_output)

        return last_hidden_state, pooled_output, final_feat


class FuseEncoder(nn.Module): 
    def __init__(self, config, d_visions, d_texts, ft_out, vision_bodies, text_bodies, vision_head, text_head, manifold:Union[CustomLorentz, PoincareBall]=None ,mapper=None, use_fused_features=True) -> None:
        super().__init__()
        self.manifold = manifold
        self.vision_bodies = nn.ModuleList(vision_bodies) 
        self.text_bodies = nn.ModuleList(text_bodies) 
        self.vision_head = vision_head 
        self.text_head = text_head
        self.config = config
        self.mapper = mapper
        self.dropout_text = nn.Dropout(0.1)
        self.dropout_vision = nn.Dropout(0.1)
        head_config = PerceiverConfig(
            d_latents=ft_out, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            self_attention_widening_factor=4,
            cross_attention_widening_factor=4,
            num_self_attention_heads=config.num_self_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        text_sizes= 0 
        vision_sizes= 0 
        for i in range(len(d_visions)): 
            text_sizes += d_texts[i]
            vision_sizes += d_visions[i]

        if use_fused_features:
            self.perceiver_head = FuseMultiModalModel(
                config=head_config, 
                d_visions=d_visions,
                d_texts=d_texts,
                num_blocks=config.num_blocks
            ) 
            # text_sizes= config.d_latents * config.num_latents * len(self.text_bodies)* config.num_blocks 
            # vision_sizes=  config.d_latents  * config.num_latents * len(self.vision_bodies) * config.num_blocks
            self.vision_fuse_proj= nn.Linear(vision_sizes, ft_out, bias=False)
            self.text_fuse_proj= nn.Linear(text_sizes , ft_out, bias=False)
            # self.vision_perceiver_proj= nn.Linear(config.d_latents, ft_out, bias=False)
            # self.text_perceiver_proj= nn.Linear(config.d_latents , ft_out, bias=False)
        else:
            self.perceiver_head = None 
            self.vision_fuse_proj= nn.Linear(vision_sizes, ft_out, bias=False)
            self.text_fuse_proj= nn.Linear(text_sizes , ft_out, bias=False)


        if self.mapper is not None:
            self.itm_head = nn.Sequential(
                LorentzLinear(manifold, ft_out + 1, [config.d_latents*2 + 1, config.d_latents*2 + 1, config.d_latents+1], dropout=0.1, act_func='gelu'),
                LorentzMLR(manifold ,config.d_latents + 1, 2) 
            )
        else:
            self.itm_head = nn.Sequential(
                SeqLinear(ft_out, [config.d_latents*2, config.d_latents*2, config.d_latents], dropout=0.1, act_func='gelu'),
                nn.Linear(config.d_latents, 2) 
            )
    

    def forward(
            self,
            pixel_values: List[torch.FloatTensor] = None,
            input_ids: List[torch.Tensor] = None,
            attention_masks: List[torch.Tensor] = None,
            num_hidden_states: [torch.Tensor] = 1,
    ) -> torch.FloatTensor:
        last_hidden_states = []
        pooled_outputs = []
        ori_feat = None
        cross_latents = None
        if pixel_values is not None:
            for i in range(len(pixel_values)):
                if i==0:
                    with torch.no_grad():
                        last_hidden_state, pooled_output, ori_embed = self.vision_bodies[i](
                            pixel_values=pixel_values[i]
                        )
                        last_hidden_states.append(last_hidden_state)
                        pooled_outputs.append(pooled_output)
                        ori_feat = ori_embed 
                else:
                    last_hidden_state, pooled_output, ori_embed = self.vision_bodies[i](
                        pixel_values=pixel_values[i],
                        num_hidden_states=num_hidden_states,
                        use_first_layers=self.config.use_first_layers
                    )
                    last_hidden_states.append(last_hidden_state)
                    pooled_outputs.append(pooled_output)

            if self.perceiver_head is not None:
                output = torch.cat(pooled_outputs, dim=-1)
                output = self.vision_fuse_proj(output)
                latents_output, cross_latents = self.perceiver_head.get_vision_features(vision_inputs=last_hidden_states, vision_oris=output)
                output = latents_output[:,0,:]
                
            else: 
                output = torch.cat(pooled_outputs, dim=-1)
                output = self.vision_fuse_proj(output)
                cross_latents = output.unsqueeze(1) 
              
        else:
            for i in range(len(input_ids)):
                if i == 0:
                    with torch.no_grad():
                        last_hidden_state, pooled_output, ori_embed = self.text_bodies[i](
                            input_ids=input_ids[i], 
                            attention_mask=attention_masks[i],
                        )
                        last_hidden_states.append(last_hidden_state)
                        pooled_outputs.append(pooled_output)
                        ori_feat = ori_embed 
                else:
                    last_hidden_state, pooled_output, ori_embed = self.text_bodies[i](
                        input_ids=input_ids[i], 
                        attention_mask=attention_masks[i],
                        num_hidden_states=num_hidden_states,
                        use_first_layers=self.config.use_first_layers
                    )
                    last_hidden_states.append(last_hidden_state)
                    pooled_outputs.append(pooled_output)
            
            # if self.perceiver_head is not None:
                # output = torch.cat(pooled_outputs, dim=-1)
                # output = self.text_fuse_proj(output)
                # latents_output, cross_latents = self.perceiver_head.get_text_features(text_inputs=last_hidden_states, text_oris=output, attention_masks=attention_masks)
                # output = torch.mean(latents_output,dim=1) + output
            # else: 
            output = torch.cat(pooled_outputs, dim=-1)
            output = self.text_fuse_proj(output)
            cross_latents = output.unsqueeze(1) 

        if self.mapper is not None:
            ori_feat = self.mapper(ori_feat, use_normalized=True)
            output = self.mapper(output, use_normalized=False)

        return cross_latents, output, ori_feat

    def compute_itm(self, vision_latents:torch.Tensor, text_latents:torch.Tensor): 
        itm_score = torch.zeros(vision_latents.shape[0], 2)
        if self.perceiver_head is not None:
            itm = self.perceiver_head.compute_itm(
                text_latents=text_latents, 
                vision_latents=vision_latents, 
            ) 
            if self.mapper is not None:
                itm = self.mapper(itm)

            itm_score = self.itm_head(itm[:,0,:])
        return itm_score 







 

