
import torch
import torch.nn as nn
from typing import Optional
from .graphs import GraphHead
from transformers import CLIPModel


class CLIPText(nn.Module): 
    def __init__(self, config ,body, head, num_trainable_blocks=0, freeze_embeddings=True) -> None:
        super().__init__()

        self.body = body
        self.head = head 
        self.config = config
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        last_hidden_state = text_outputs[0]

        if not self.config.use_lorentz_centroid or self.config.manifold != 'lorentz':
            pooled_output = text_outputs[1]
        else:
            pooled_output = last_hidden_state
        for layer in self.head:
                pooled_output = layer(pooled_output)


        return last_hidden_state, pooled_output
    
class CLIPGraphText(nn.Module): 
    def __init__(self, config ,body, head, manifold_mapper=None, num_layers=1) -> None:
        super().__init__()
        self.body = body
        self.head = head 
        self.manifold_mapper = manifold_mapper
        
        self.graph_head = GraphHead(
            sizes=[512] * num_layers, 
            proj_hidden_sizes=[512, 512, 512], 
            ft_out=512
        ) 
        self.config = config
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

        pooled_output = text_outputs[1]
        last_hidden_state = text_outputs[0]

        pooled_output = self.head(pooled_output)
        output, graph_output = self.graph_head(hidden_states=text_outputs.hidden_states, pooled_output=pooled_output)
        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)

        return last_hidden_state, output, graph_output

class BLIPGraphText(nn.Module): 
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        blip_text_outputs = self.blip_body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        # clip_text_outputs = self.clip_body(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     return_dict=True,
        #     output_hidden_states=True,
        # )

        last_hidden_state = blip_text_outputs[0]
        blip_pooled_output = last_hidden_state[:, 0 ,:]
        blip_pooled_output = self.blip_head(blip_pooled_output)

        output, graph_output = self.graph_head(hidden_states=blip_text_outputs.hidden_states, pooled_output=blip_pooled_output)

        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)


        return last_hidden_state, output, graph_output

        
class BLIPText(nn.Module): 
    def __init__(self, config ,body, head) -> None:
        super().__init__()

        self.body = body
        self.head = head 
        self.config = config
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        last_hidden_state = text_outputs[0]
        if not self.config.use_lorentz_centroid or self.config.manifold != 'lorentz':
            pooled_output = last_hidden_state[:, 0, :]
        else:
            pooled_output = last_hidden_state

        for layer in self.head:
            pooled_output = layer(pooled_output)


        return last_hidden_state, pooled_output



class LavisBLIPText(nn.Module): 
    def __init__(self, config, body, head ) -> None:
        super().__init__()

        self.body = body
        self.head = head 
        self.config = config

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden_state = text_outputs[0]
        if not self.config.use_lorentz_centroid or self.config.manifold != 'lorentz':
            pooled_output = last_hidden_state[:, 0, :]
        else:
            pooled_output = last_hidden_state

        for layer in self.head:
            pooled_output = layer(pooled_output)


        return last_hidden_state, pooled_output
     
    
