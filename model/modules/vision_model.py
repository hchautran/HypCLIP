
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from .seq_linear import LorentzSeqLinear 
from .graphs import GraphHead

class CLIPVision(nn.Module): 
    def __init__(self,config, body, head, num_trainable_blocks=0, freeze_embedding=True) -> None:
        super().__init__()

        self.body = body
        self.head = head 
        self.config = config


    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        vision_outputs = self.body(
            pixel_values=pixel_values,
            return_dict=True,
        )

        last_hidden_state = vision_outputs[0]

        if not self.config.use_lorentz_centroid or self.config.manifold != 'lorentz':
            pooled_output = vision_outputs[1]
        else:
            pooled_output = last_hidden_state
        for layer in self.head:
            pooled_output = layer(pooled_output)

        return last_hidden_state, pooled_output
    
class CLIPGraphVision(nn.Module): 
    def __init__(self, config , body, head, manifold_mapper=None, num_layers=1) -> None:
        super().__init__()
        self.body = body
        self.head = head 
        self.manifold_mapper = manifold_mapper
        self.graph_head = GraphHead(
            sizes=[768] * num_layers, 
            proj_hidden_sizes=[512, 512, 512], 
            ft_out=512
        ) 
        self.config = config
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        vision_outputs = self.body(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        pooled_output = vision_outputs[1]
        last_hidden_state = vision_outputs[0]

        pooled_output = self.head(pooled_output)
        # output = torch.cat([self.graph_head(hidden_states=vision_outputs.hidden_states, pooled_output=pooled_output), pooled_output], dim=-1)
        # output = self.final_proj(self.dropout(output))
        output, graph_output = self.graph_head(hidden_states=vision_outputs.hidden_states, pooled_output=pooled_output)
        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)

        return last_hidden_state, output, graph_output



class BLIPGraphVision(nn.Module): 
    def __init__(self, config, blip_body, blip_head, clip_body, manifold_mapper=None, num_layers=1) -> None:
        super().__init__()
        self.blip_body = blip_body
        self.clip_body = clip_body
        self.blip_head = blip_head 
        self.manifold_mapper = manifold_mapper

        self.graph_head = GraphHead(
            sizes=[768] * num_layers, 
            proj_hidden_sizes=[512, 512, 256], 
            graph_heads=2, 
            graphs_hidden_channel=256, 
            ft_out=256
        ) 
        self.config = config
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        blip_vision_outputs = self.blip_body(
            pixel_values=pixel_values,
            return_dict=True,
            output_hidden_states=True,
        )
        # clip_vision_outputs = self.clip_body(
        #     pixel_values=pixel_values,
        #     return_dict=True,
        #     output_hidden_states=True,
        # )

        last_hidden_state = blip_vision_outputs[0]
        blip_pooled_output = last_hidden_state[:, 0 ,:]
        blip_pooled_output = self.blip_head(blip_pooled_output)

        output, graph_output = self.graph_head(hidden_states=blip_vision_outputs.hidden_states, pooled_output=blip_pooled_output)

        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)

        return last_hidden_state, output, graph_output

class BLIPVision(nn.Module): 
    def __init__(self, config, body, head ) -> None:
        super().__init__()

        self.body = body
        self.head = head 
        self.config = config

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        vision_outputs = self.body(
            pixel_values=pixel_values,
            return_dict=False,
        )

        last_hidden_state = vision_outputs[0]

        pooled_output = last_hidden_state[:, 0, :]
        for layer in self.head:
            pooled_output = layer(pooled_output)

        return last_hidden_state, pooled_output
    

     


