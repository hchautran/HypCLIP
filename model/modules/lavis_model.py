
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from .seq_linear import LorentzSeqLinear 
from .graphs import GraphHead, LorentzGraphHead
from hyptorch.lorentz.manifold import CustomLorentz


class Text(object):
    pass

class LavisEncoder(nn.Module): 
    def __init__(self, config, body, head, mapper=None, use_normalized=False ) -> None:
        super().__init__()
        self.body = body
        self.head = head 
        self.config = config
        self.mapper = mapper
        self.use_normalized = use_normalized

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is not None:
            with torch.no_grad():
                outputs = self.body.forward_features(
                    pixel_values,
                )
                last_hidden_state = outputs
                pooled_output = last_hidden_state[:, 0, :]
                pooled_output = self.head(pooled_output)
                if self.mapper is not None:
                    pooled_output = self.mapper(pooled_output, use_normalized=True)
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.body.forward_text(text)

            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.head(pooled_output)
            if self.mapper is not None:
                pooled_output = self.mapper(pooled_output, use_normalized=self.use_normalized)


        return last_hidden_state, pooled_output


class LavisBLIPGraphHead(nn.Module): 
    def __init__(self, ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, graph_hidden_channels=512) -> None:
        super().__init__()
        self.config = config
        self.body = body
        self.head = head 
        hidden_sizes = [hidden_size] * num_hidden_layers + [ft_out] 
        self.manifold_mapper = manifold_mapper
        self.graph_head = GraphHead(
            sizes=[ft_in] * num_layers, 
            proj_hidden_sizes=hidden_sizes, 
            graphs_hidden_channel=graph_hidden_channels,
            ft_out=ft_out
        ) 
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        if pixel_values is not None:
            outputs = self.body.forward_features(pixel_values)
            last_hidden_state = outputs
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.body.forward_text(text)
            last_hidden_state = outputs.last_hidden_state

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.head(pooled_output)

        output, graph_output = self.graph_head(hidden_states=[last_hidden_state], pooled_output=pooled_output)
        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)

        return last_hidden_state, output, graph_output


class LavisLorentzBLIPGraphHead(nn.Module): 
    def __init__(self, manifold:CustomLorentz ,ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, graph_hidden_channels=512) -> None:
        super().__init__()
        self.config = config
        self.body = body
        self.head = head 
        self.manifold = manifold
        hidden_sizes = [hidden_size] * num_hidden_layers + [ft_out] 
        self.manifold_mapper = manifold_mapper
        self.graph_head = LorentzGraphHead(
            manifold=manifold,
            sizes=[ft_in] * num_layers, 
            proj_hidden_sizes=hidden_sizes, 
            graphs_hidden_channel=graph_hidden_channels, 
            ft_out=ft_out,
            dropout_edge_ratio=0.5,
        ) 
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        if pixel_values is not None:
            outputs = self.body.forward_features(
                pixel_values,
            )
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.head(pooled_output)
            if self.manifold_mapper is not None:
                pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
                lorentz_hidden_states = [self.manifold_mapper(last_hidden_state)]
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.body.forward_text(text)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.head(pooled_output)

            if self.manifold_mapper is not None:
                pooled_output = self.manifold_mapper(pooled_output, use_normalized=False)
                lorentz_hidden_states = [self.manifold_mapper(last_hidden_state)]


        output, graph_output = self.graph_head(hidden_states=lorentz_hidden_states, pooled_output=pooled_output)

        return last_hidden_state, output, graph_output
        
 