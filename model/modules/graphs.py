import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .seq_linear import SeqLinear, LorentzSeqLinear
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .HypGAT import LorentzGAT
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers import LorentzLinear, LorentzAct
from torch_geometric.utils import dropout_edge 
from lavis import BlipRetrieval
class ProjLayers(nn.Module):
  def __init__(self,  sizes=[768], hidden_sizes=[512],  dropout=0.1, shared=False):
    super().__init__()
    self.shared = shared
    if not shared:
        self.projectors = nn.ModuleList([SeqLinear(ft_in=size, layer_dims=hidden_sizes, dropout=dropout, act_func='gelu') for size in sizes])
    else:
        self.projector = SeqLinear(ft_in=sizes[-1], layer_dims=hidden_sizes, dropout=dropout, act_func='gelu')
    self.layer_norm = nn.LayerNorm(hidden_sizes[-1])

  def forward(self, hidden_states:torch.Tensor):
    outputs = []
    for i in range(len(hidden_states)):
        if not self.shared:
            output= self.projectors[i](hidden_states[i])
        else:
            output= self.projector(hidden_states[i])
        outputs.append(self.layer_norm(output))

    return outputs

class GraphHead(nn.Module):
    def __init__(self, sizes=[768], proj_hidden_sizes=[512, 512], ft_out=512 ,dropout=0.1, graph_heads=4, graphs_hidden_channel=512, dropout_edge_ratio=0.1, shared=False, gamma=0.5):
        super().__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.proj_layers = ProjLayers(sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.gnn = GNN(ft_in=proj_hidden_sizes[-1], hidden_channels=graphs_hidden_channel, num_heads=graph_heads ,ft_out=ft_out) 
        self.final_proj = SeqLinear(ft_in=ft_out*2, layer_dims=[512 , ft_out], dropout=dropout, act_func='gelu')
        # self.batch_norm = nn.BatchNorm1d(proj_hidden_sizes[-1])
        self.dropout_edge_ratio =  dropout_edge_ratio

    def forward(self, hidden_states:torch.Tensor, pooled_output:torch.Tensor):
        ends = []
        starts = []
        begin_index = 1
        hidden_states = hidden_states[(len(hidden_states) - self.num_layers):] 
        output = self.proj_layers(hidden_states)
        

        output = torch.cat(self.proj_layers(hidden_states), dim =-2)
        bs = output.shape[0] 

        output = torch.cat([pooled_output.view(bs, 1, -1), output], dim=-2)
        for i in range(len(hidden_states)):
            for j in range(hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
        begin_index += hidden_states[i].shape[1]
        edge_index = torch.tensor([starts, ends], dtype=torch.long).to(output.get_device())
        edge_index = add_self_loops(edge_index)[0]
        graphs = []
        for i in range(bs):
            graphs.append(Data(x=output[i,:,:], edge_index=edge_index))
        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        graph_output, graph_mean = self.gnn(data_batch, batch_size=bs)
        output = graph_output + pooled_output
        return output, graph_mean 

class GNN(torch.nn.Module):
    def __init__(self, ft_in ,hidden_channels, ft_out, num_heads=4):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(ft_in, hidden_channels//num_heads, dropout=0.6, heads=num_heads, concat=True)  
        self.act1 = nn.GELU()
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels//num_heads, dropout=0.6, heads=num_heads, concat=True)
        # self.act2 = nn.GELU()
        # self.conv3 = GATv2Conv(hidden_channels, hidden_channels, dropout=0.4, heads=1, concat=False)
        self.lin = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels*4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=hidden_channels*4 , out_features=hidden_channels),
        )
        self.final_lin = nn.Linear(hidden_channels, ft_out)

    def forward(self, graphs, batch_size):
        x, edge_index,batch = graphs.x, graphs.edge_index, graphs.batch
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        # x = self.act2(x)
        # x = self.conv3(x, edge_index)
        graph_mean = global_mean_pool(x, batch)
        x = x.view(batch_size, graphs.x.shape[0]//batch_size, -1)
        x = x[:, 0, :]
        x= self.lin(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.final_lin(x)
        return x, graph_mean

class GraphModel(nn.Module): 
    def __init__(self, ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, shared_proj_layers=False, graph_hidden_channels=512) -> None:
        super().__init__()
        self.config = config
        self.body = body
        self.head = head 
        hidden_sizes = [hidden_size] * num_hidden_layers + [ft_out] 
        self.manifold_mapper = manifold_mapper
        self.graph_head = GraphHead(
            sizes=[ft_in] * num_layers, 
            proj_hidden_sizes=hidden_sizes, 
            ft_out=ft_out,
            graphs_hidden_channel=graph_hidden_channels,
            dropout_edge_ratio=0.0,
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
            outputs = self.body(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )

     
        last_hidden_state = outputs[0]
        if 'blip' in self.config.model_ckt:
            pooled_output = last_hidden_state[:, 0, :]
        else:
            pooled_output = outputs[1]

        pooled_output = self.head(pooled_output)

        
        output, graph_output = self.graph_head(hidden_states=outputs.hidden_states, pooled_output=pooled_output)
        if self.manifold_mapper is not None:
            output = self.manifold_mapper(output)
            graph_output = self.manifold_mapper(graph_output)

        return last_hidden_state, output, graph_output

class LorentzGNN(torch.nn.Module):
    def __init__(self, manifold:CustomLorentz ,ft_in ,hidden_channels, ft_out):
        super(LorentzGNN, self).__init__()
        torch.manual_seed(12345)
        self.manifold = manifold
        self.conv1 = GATv2Conv(ft_in, hidden_channels//4, heads=4 ,dropout=0.6)  
        # self.conv1 = LorentzGAT(manifold, ft_in, hidden_channels, dropout=0.5)
        self.act1 = nn.GELU()
        # self.act1 = LorentzAct(nn.GELU(), manifold=manifold)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels//4, heads=4 ,dropout=0.6)
        # self.conv2 = LorentzGAT(manifold, hidden_channels, hidden_channels, dropout=0.5)
        # self.act2 = LorentzAct(nn.GELU(), manifold=manifold)
        # self.conv3 = LorentzGAT(manifold, hidden_channels, hidden_channels, dropout=0.5)
        self.lin = nn.Sequential(
            LorentzLinear(manifold=manifold, in_features=hidden_channels + 1, out_features=hidden_channels*4 + 1, dropout=0.3),
            LorentzAct(nn.GELU(), manifold=manifold),
            LorentzLinear(manifold=manifold, in_features=hidden_channels*4 + 1, out_features=hidden_channels + 1, dropout=0.3),
        )
        self.final_lin = LorentzLinear(manifold=manifold, in_features=hidden_channels+ 1, out_features=ft_out + 1, dropout=0.1)

    def forward(self, graphs, batch_size):
        x, edge_index, _ = graphs.x, graphs.edge_index, graphs.batch
        x = self.manifold.get_space(x)
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.manifold.add_time(x)
        x = x.view(batch_size, graphs.x.shape[0]//batch_size, -1)
        graph_mean = self.manifold.centroid(x=x) 
        x = x[:, 0, :]
        x = self.lin(x)
        # self.manifold.assert_check_point_on_manifold(x)
        # print(graph_mean)
        x = self.final_lin(x)
        self.manifold.assert_check_point_on_manifold(x)
        return x, graph_mean

class LorentzProjLayers(nn.Module):
  def __init__(self, manifold:CustomLorentz ,sizes=[768], hidden_sizes=[512],  dropout=0.1, shared=False):
    super().__init__()
    self.shared = shared
    self.sizes = sizes
    self.manifold = manifold
    if not shared:
        self.projectors = nn.ModuleList([
            LorentzSeqLinear(
                manifold=manifold,
                ft_in=size+1, 
                layer_dims=[hidden_size+1 for hidden_size in hidden_sizes], 
                dropout=dropout, 
                act_func='gelu'
        ) for size in sizes])
    else:
        self.projector = LorentzSeqLinear(
            manifold=manifold,
            ft_in=sizes[0]+1, 
            layer_dims=[hidden_size+1 for hidden_size in hidden_sizes], 
            dropout=dropout, 
            act_func='gelu'
        )  
    self.layer_norm = nn.LayerNorm(hidden_sizes[-1])

  def forward(self, hidden_states:torch.Tensor):
    outputs = []
    for i in range(len(hidden_states)):
        if not self.shared:
            output = self.projectors[i](hidden_states[i])
        else:
            output = self.projector(hidden_states[i])
        output_space = self.manifold.get_space(output)
        output_space = self.layer_norm(output_space)
        outputs.append(self.manifold.add_time(output_space))

    return outputs

class LorentzGraphHead(nn.Module):
    def __init__(self, manifold:CustomLorentz ,sizes=[768], proj_hidden_sizes=[512, 512], ft_out=512 ,dropout=0.1, graphs_hidden_channel=256, dropout_edge_ratio=0.1, shared=False, gamma=0.5):
        super().__init__()
        self.sizes = sizes
        self.manifold = manifold
        self.num_layers = len(sizes)
        self.proj_layers = LorentzProjLayers(manifold=manifold, sizes=sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.gnn = LorentzGNN(manifold=manifold, ft_in=proj_hidden_sizes[-1], hidden_channels=graphs_hidden_channel, ft_out=ft_out) 
        # self.final_proj = LorentzSeqLinear(manifold, ft_in=ft_out*2 + 1, layer_dims=[513 ,ft_out + 1], dropout=dropout, act_func='gelu')
        self.dropout_edge_ratio = dropout_edge_ratio

    def forward(self, hidden_states:torch.Tensor, pooled_output:torch.Tensor):

        ends = []
        starts = []
        begin_index = 1
        hidden_states = hidden_states[(len(hidden_states) - self.num_layers):] 
        output = self.proj_layers(hidden_states)
        
        output = torch.cat(self.proj_layers(hidden_states), dim =-2)
        bs = output.shape[0] 

        output = torch.cat([pooled_output.view(bs, 1, -1), output], dim=-2)
        # self.manifold.assert_check_point_on_manifold(output)

        for i in range(len(hidden_states)):
            for j in range(hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
        begin_index += hidden_states[i].shape[1]
        edge_index = torch.tensor([starts, ends], dtype=torch.long).to(output.get_device())
        edge_index = add_self_loops(edge_index)[0]

        graphs = []
        for i in range(bs):
            graphs.append(Data(x=output[i,:,:], edge_index=edge_index))
        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        graph_output, graph_mean = self.gnn(data_batch, batch_size=bs)

        output = self.manifold.get_space(graph_output) + self.manifold.get_space(pooled_output)
        output = self.manifold.add_time(output)

        self.manifold.assert_check_point_on_manifold(output)
        return output, graph_mean


class LorentzGraphModel(nn.Module): 
    def __init__(self, manifold:CustomLorentz ,ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, shared_proj_layers=False, graph_hidden_channels=512) -> None:
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
            ft_out=ft_out,
            graphs_hidden_channel=graph_hidden_channels,
            dropout_edge_ratio=0.5,
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
        lorentz_hidden_states = []

        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_state = outputs[0]
            if 'blip' in self.config.model_ckt:
                pooled_output = last_hidden_state[:, 0, :]
            else:
                pooled_output = outputs[1]
            pooled_output = self.head(pooled_output)
            if self.manifold_mapper is not None:
                pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
                for hidden_state in outputs.hidden_states:
                    lorentz_hidden_states.append(self.manifold_mapper(hidden_state))
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs[0]
            if 'blip' in self.config.model_ckt:
                pooled_output = last_hidden_state[:, 0, :]
            else:
                pooled_output = outputs[1]
            pooled_output = self.head(pooled_output)
            if self.manifold_mapper is not None:
                pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
                for hidden_state in outputs.hidden_states:
                    lorentz_hidden_states.append(self.manifold_mapper(hidden_state))

        output, graph_output = self.graph_head(hidden_states=lorentz_hidden_states, pooled_output=pooled_output)

        return last_hidden_state, output, graph_output

