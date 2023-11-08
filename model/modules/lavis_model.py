
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from .seq_linear import LorentzSeqLinear 
from .graphs import GraphHead, LorentzGraphHead
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .seq_linear import LorentzSeqLinear
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from torch_geometric.utils import dropout_edge 
from .graphs import LorentzProjLayers, LorentzGNN

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
            # with torch.no_grad():
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
                pooled_output = self.mapper(pooled_output, use_normalized=True)


        return last_hidden_state, pooled_output


class LavisBLIPGraphHead(nn.Module): 
    def __init__(self, ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, graph_hidden_channels=512, use_root=True) -> None:
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
            ft_out=ft_out,
            use_root=use_root
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
    def __init__(self, manifold:CustomLorentz ,ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, graph_hidden_channels=512, use_root=True) -> None:
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
            dropout=0.3,
            dropout_edge_ratio=0.5,
            use_root=use_root
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
                pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
                lorentz_hidden_states = [self.manifold_mapper(last_hidden_state)]


        output, graph_output = self.graph_head(hidden_states=lorentz_hidden_states, pooled_output=pooled_output)

        return last_hidden_state, output, graph_output
        
 

class LorentzFuseGraphHead(nn.Module):
    def __init__(
        self, 
        manifold:CustomLorentz ,
        sizes_1=[768], 
        sizes_2=[512],
        proj_hidden_sizes=[512, 512], 
        ft_out_1=256,
        ft_out_2=512,
        dropout=0.1, 
        graphs_hidden_channel=256, 
        dropout_edge_ratio=0.1, 
        shared=False
    ):
        super().__init__()
        self.sizes_1 = sizes_1
        self.sizes_2 = sizes_2
        self.manifold = manifold
        self.num_layers_1 = len(sizes_1)
        self.num_layers_2 = len(sizes_2)
        self.proj_layers_1 = LorentzProjLayers(manifold=manifold, sizes=sizes_1, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.proj_layers_2 = LorentzProjLayers(manifold=manifold, sizes=sizes_2, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.gnn = LorentzGNN(manifold=manifold, ft_in=proj_hidden_sizes[-1], hidden_channels=graphs_hidden_channel, ft_out=ft_out_1) 
        self.final_proj = LorentzSeqLinear(manifold, ft_in=(ft_out_1*2 + ft_out_2 + 1), layer_dims=[1024, ft_out_2], dropout=dropout, act_func='gelu')
        self.dropout_edge_ratio = dropout_edge_ratio
    
    def build_graph_edge(self, hidden_states_1:torch.Tensor, hidden_states_2:torch.Tensor ,batch_size:int ,data:torch.Tensor):

        ends = []
        starts = []
        begin_index = 1
        for i in range(len(hidden_states_1)):
            for j in range(hidden_states_1[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
        begin_index += hidden_states_1[i].shape[1]

        for i in range(len(hidden_states_2)):
            for j in range(hidden_states_2[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
        begin_index += hidden_states_2[i].shape[1]

        edge_index = torch.tensor([starts, ends], dtype=torch.long).to(data.get_device())

        edge_index = add_self_loops(edge_index)[0]

        graphs = []
        for i in range(batch_size):
            graphs.append(Data(x=data[i,:,:], edge_index=edge_index))
        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        return data_batch


    def forward(self, hidden_states_1:torch.Tensor, hidden_states_2:torch.Tensor, pooled_output:torch.Tensor):

        bs = pooled_output.shape[0] 
        hidden_states_1 = hidden_states_1[(len(hidden_states_1) - self.num_layers_1):] 
        hidden_states_2 = hidden_states_2[(len(hidden_states_2) - self.num_layers_2):] 
        
        output_1 = torch.cat(self.proj_layers_1(hidden_states_1), dim =-2)
        output_2 = torch.cat(self.proj_layers_2(hidden_states_2), dim =-2)

        output = torch.cat([pooled_output.view(bs, 1, -1), output_1, output_2], dim=-2)
        # self.manifold.assert_check_point_on_manifold(output)
        data_batch = self.build_graph_edge(
            hidden_states_1=hidden_states_1, 
            hidden_states_2=hidden_states_2, 
            batch_size=bs, 
            data=output
        )

        graph_output, graph_mean = self.gnn(data_batch, batch_size=bs)
        output = self.manifold.get_space(graph_output) + self.manifold.get_space(pooled_output)
        output = self.manifold.add_time(output)

        self.manifold.assert_check_point_on_manifold(output)
        return output, graph_mean

 
class FuseLorentzGraphModel(nn.Module): 
    def __init__(
        self, 
        manifold:CustomLorentz,
        config, 
        ft_in_1, 
        ft_in_2, 
        ft_out, 
        body_1, 
        head_1, 
        body_2, 
        head_2, 
        manifold_mapper=None, 
        num_layers_1=1, 
        num_layers_2=1, 
        hidden_size=512, 
        num_hidden_layers=2, 
        shared_proj_layers=False
    ) -> None:
        super().__init__()
        self.config = config
        self.body_1 = body_1
        self.head_1 = head_1
        self.body_2 = body_2
        self.head_2 = head_2
        self.manifold = manifold
        self.manifold_mapper = manifold_mapper
        self.graph_head = LorentzFuseGraphHead(
            manifold=manifold,
            sizes_1=[ft_in_1] * num_layers_1, 
            sizes_2=[ft_in_2] * num_layers_2, 
            proj_hidden_sizes_1=[hidden_size] * num_hidden_layers + [ft_out], 
            proj_hidden_sizes_2=[hidden_size] * num_hidden_layers + [ft_out], 
            ft_out=ft_out,
            graphs_hidden_channel=512,
            dropout_edge_ratio=0.7,
            dropout=0.3,
            shared=shared_proj_layers
        ) 
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        if pixel_values is not None:
            outputs_1 = self.body_1(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            outputs_2 = self.body_2(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs_1 = self.body_1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            outputs_2 = self.body_2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )

        last_hidden_state_1 = outputs_1[0]

        pooled_output_1 = last_hidden_state_1[:, 0, :]
        pooled_output_2 = outputs_2[1]
            
        hidden_states_1 = []
        hidden_states_2 = []

        pooled_output_1 = self.head_1(pooled_output_1)
        pooled_output_2 = self.head_2(pooled_output_2)
        
        if self.manifold_mapper is not None:
            pooled_output_1 = self.manifold_mapper(pooled_output_1)
            pooled_output_2 = self.manifold_mapper(pooled_output_2)
            for hidden_state in outputs_1.hidden_states:
                hidden_states_1.append(self.manifold_mapper(hidden_state))
            for hidden_state in outputs_2.hidden_states:
                hidden_states_2.append(self.manifold_mapper(hidden_state))

        # print(lorentz_hidden_states)

        output, graph_output = self.graph_head(hidden_states_1=hidden_states_1, hidden_states_2=hidden_states_2 ,pooled_output_1=pooled_output_1, pooled_output_2=pooled_output_2)

        return last_hidden_state_1, output, graph_output


