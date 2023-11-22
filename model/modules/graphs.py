import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .seq_linear import SeqLinear, LorentzSeqLinear
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .HypGAT import LorentzGAT
from hyptorch.geoopt import ManifoldParameter
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
    def __init__(self, text_sizes=[768] ,vision_sizes=[768], proj_hidden_sizes=[512, 512], ft_out=512 ,dropout=0.1, graphs_hidden_channel=256, dropout_edge_ratio=0.1, shared=False, use_root=False):
        super().__init__()
        self.text_sizes = text_sizes 
        self.vision_sizes = vision_sizes 
        self.num_text_layers = len(text_sizes)
        self.num_vision_layers = len(vision_sizes)
        self.vision_proj_layers = ProjLayers(sizes=vision_sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.text_proj_layers = ProjLayers(sizes=text_sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.gnn = GNN(ft_in=proj_hidden_sizes[-1], hidden_channels=graphs_hidden_channel, ft_out=ft_out) 
        if use_root:
            self.root = nn.Parameter(data=torch.zeros((1,proj_hidden_sizes[-1]), requires_grad=True))
        else:
            self.root = None 
        # self.final_proj = LorentzSeqLinear(manifold, ft_in=ft_out*2 + 1, layer_dims=[513 ,ft_out + 1], dropout=dropout, act_func='gelu')
        self.dropout_edge_ratio = dropout_edge_ratio
    
    def add_root(self, graph, bs):
        starts = []
        ends = []
        if self.root is not None:
            graph_size = graph.x.shape[0] // bs 
            graph.x = torch.cat([graph.x, self.root])
            for i in range(bs):
                starts.append(graph.x.shape[0]-1)
                starts.append(i * graph_size)
                ends.append(i * graph_size)
                ends.append(graph.x.shape[0]-1)
            graph.edge_index = torch.cat([graph.edge_index, torch.tensor([starts, ends], dtype=torch.long).to(graph.x.get_device())], dim=-1)
        return graph
    
    def build_graph(self, hidden_states, x):
        bs = x.shape[0] 
        ends = []
        starts = []
        begin_index = 1
        for i in range(len(hidden_states)):
            for j in range(hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += hidden_states[i].shape[1]
        edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x.get_device())
        edge_index = add_self_loops(edge_index)[0]
        graphs = []
        for i in range(bs):
            graphs.append(Data(x=x[i,:,:], edge_index=edge_index))

        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        return data_batch
        
    def build_graph_itm(self, text_hidden_states, vision_hidden_states, x_text, x_vision):
        bs = x_text.shape[0] 
        ends = []
        starts = []

        begin_index = 1
        for i in range(len(text_hidden_states)):
            for j in range(text_hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += text_hidden_states[i].shape[1]

        text_edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x_text.get_device())
        text_edge_index = add_self_loops(text_edge_index)[0]

        ends = []
        starts = []
        for i in range(len(vision_hidden_states)):
            for j in range(vision_hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += vision_hidden_states[i].shape[1]
        vision_edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x_text.get_device())
        vision_edge_index= add_self_loops(vision_edge_index)[0]

        graphs = []
        for i in range(bs):
            graphs.append(Data(x=x_text[i,:,:], edge_index=text_edge_index))
        for i in range(bs):
            graphs.append(Data(x=x_vision[i,:,:], edge_index=vision_edge_index))

        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        return data_batch

    def forward(self, hidden_states:torch.Tensor, pooled_output:torch.Tensor, mode='text'):
        if mode == 'text':
            hidden_states = hidden_states[(len(hidden_states) - self.num_text_layers):] 
            output = torch.cat(self.text_proj_layers(hidden_states), dim =-2)
        else:
            hidden_states = hidden_states[(len(hidden_states) - self.num_vision_layers):] 
            output = torch.cat(self.vision_proj_layers(hidden_states), dim =-2)

        bs = output.shape[0] 
        output = torch.cat([pooled_output.view(bs, 1, -1), output], dim=-2)
        graph = self.build_graph(hidden_states=hidden_states, x=output) 
        graph = self.add_root(graph=graph, bs=bs) 
        graph_output, mean = self.gnn(graph, batch_size=bs, use_root=(self.root is not None))
        output = graph_output + pooled_output

        self.manifold.assert_check_point_on_manifold(output)
        return output, mean
    


class GNN(torch.nn.Module):
    def __init__(self, ft_in ,hidden_channels, ft_out, num_heads=4):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(ft_in, hidden_channels//num_heads, dropout=0.7, heads=num_heads, concat=True)  
        self.act1 = nn.GELU()
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels//num_heads, dropout=0.7, heads=num_heads, concat=True)
        self.act2 = nn.GELU()
        # self.conv3 = GATv2Conv(hidden_channels, hidden_channels, dropout=0.4, heads=1, concat=False)
        self.lin = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels*4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=hidden_channels*4 , out_features=hidden_channels),
            nn.GELU(),
        )
        self.final_lin = nn.Linear(hidden_channels, ft_out)

    def forward(self, graphs, batch_size, use_root=False):
        x, edge_index,batch = graphs.x, graphs.edge_index, graphs.batch
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        # x = self.conv3(x, edge_index)
        if use_root:
            x = x[:x.shape[0]-1]
        graph_mean = global_mean_pool(x, batch)
            
        x = x.view(batch_size, x.shape[0]//batch_size, -1)
        x = x[:, 0, :]
        x= self.lin(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.final_lin(x)
        return x, graph_mean

class GraphModel(nn.Module): 
    def __init__(self, ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, shared_proj_layers=False, graph_hidden_channels=512, use_root=False) -> None:
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
            shared=shared_proj_layers,
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
        self.conv1 = GATv2Conv(ft_in, hidden_channels//4, heads=4 ,dropout=0.7)  
        # self.conv1 = LorentzGAT(manifold, ft_in, hidden_channels, dropout=0.3)
        self.act1 = nn.GELU()
        # self.act1 = LorentzAct(nn.GELU(), manifold=manifold)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels//4, heads=4 ,dropout=0.7)
        # self.conv2 = LorentzGAT(manifold, hidden_channels, hidden_channels, dropout=0.3)
        # self.act2 = LorentzAct(nn.GELU(), manifold=manifold)
        # self.conv3 = LorentzGAT(manifold, hidden_channels, hidden_channels, dropout=0.5)
        self.lin = nn.Sequential(
            LorentzLinear(manifold=manifold, in_features=hidden_channels + 1, out_features=hidden_channels*4 + 1, dropout=0.2, bias=True),
            LorentzAct(nn.GELU(), manifold=manifold),
            LorentzLinear(manifold=manifold, in_features=hidden_channels*4 + 1, out_features=hidden_channels + 1, dropout=0.2, bias=True),
        )
        self.final_lin = LorentzLinear(manifold=manifold, in_features=hidden_channels+ 1, out_features=ft_out + 1, dropout=0.2)

    def forward(self, graphs, batch_size, use_root=False):
        x, edge_index, _ = graphs.x, graphs.edge_index, graphs.batch
        x = self.manifold.get_space(x)
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.manifold.add_time(x)
        # print(x.shape)
        
        if use_root:
            x = x[:x.shape[0]-1]
            root = x[-1]
            # print(x.shape)
        else:
            root = self.manifold.centroid(x=x) 

        x = x.view(batch_size, x.shape[0]//batch_size, -1)
        x = x[:, 0, :]
        x = self.lin(x)
        # self.manifold.assert_check_point_on_manifold(x)
        # print(graph_mean)
        x = self.final_lin(x)
        self.manifold.assert_check_point_on_manifold(x)
        return x, root 
    


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

  def forward(self, hidden_states:torch.Tensor):
    outputs = []
    for i in range(len(hidden_states)):
        if not self.shared:
            output = self.projectors[i](hidden_states[i])
        else:
            output = self.projector(hidden_states[i])
        outputs.append(output)

    return outputs

class LorentzGraphHead(nn.Module):
    def __init__(self, manifold:CustomLorentz , text_sizes=[768] ,vision_sizes=[768], proj_hidden_sizes=[512, 512], ft_out=512 ,dropout=0.1, graphs_hidden_channel=256, dropout_edge_ratio=0.1, shared=False, use_root=False):
        super().__init__()
        self.text_sizes = text_sizes 
        self.vision_sizes = vision_sizes 
        self.manifold = manifold

        self.num_text_layers = len(text_sizes)
        self.num_vision_layers = len(vision_sizes)
        
        # self.vision_proj_layers = ProjLayers(sizes=vision_sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        # self.text_proj_layers = ProjLayers(sizes=text_sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout, shared=shared)
        self.gnn = LorentzGNN(manifold=manifold, ft_in=256, hidden_channels=graphs_hidden_channel, ft_out=ft_out) 
        if use_root:
            self.root = ManifoldParameter(data=manifold.origin((1,proj_hidden_sizes[-1] + 1)), manifold=manifold, requires_grad=False) 
        else:
            self.root = None 
        # self.final_proj = LorentzSeqLinear(manifold, ft_in=ft_out*2 + 1, layer_dims=[513 ,ft_out + 1], dropout=dropout, act_func='gelu')
        self.dropout_edge_ratio = dropout_edge_ratio
    
    def add_root(self, graph, bs):
        starts = []
        ends = []
        if self.root is not None:
            graph_size = graph.x.shape[0] // bs 
            graph.x = torch.cat([graph.x, self.root])
            for i in range(bs):
                starts.append(graph.x.shape[0]-1)
                starts.append(i * graph_size)
                ends.append(i * graph_size)
                ends.append(graph.x.shape[0]-1)
            graph.edge_index = torch.cat([graph.edge_index, torch.tensor([starts, ends], dtype=torch.long).to(graph.x.get_device())], dim=-1)
        return graph
    
    def build_graph(self, hidden_states, x):
        bs = x.shape[0] 
        ends = []
        starts = []
        begin_index = 1
        for i in range(len(hidden_states)):
            for j in range(hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += hidden_states[i].shape[1]
        edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x.get_device())
        edge_index = add_self_loops(edge_index)[0]
        graphs = []
        for i in range(bs):
            graphs.append(Data(x=x[i,:,:], edge_index=edge_index))

        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        return data_batch
        
    def build_graph_itm(self, text_hidden_states, vision_hidden_states, x_text, x_vision):
        bs = x_text.shape[0] 
        ends = []
        starts = []

        begin_index = 1
        for i in range(len(text_hidden_states)):
            for j in range(text_hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += text_hidden_states[i].shape[1]

        text_edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x_text.get_device())
        text_edge_index = add_self_loops(text_edge_index)[0]

        ends = []
        starts = []
        for i in range(len(vision_hidden_states)):
            for j in range(vision_hidden_states[i].shape[-2]):
                starts.append(0)
                starts.append(begin_index + j)
                ends.append(begin_index + j)
                ends.append(0)
            begin_index += vision_hidden_states[i].shape[1]
        vision_edge_index = torch.tensor([starts, ends], dtype=torch.long).to(x_text.get_device())
        vision_edge_index= add_self_loops(vision_edge_index)[0]

        graphs = []
        for i in range(bs):
            graphs.append(Data(x=x_text[i,:,:], edge_index=text_edge_index))
        for i in range(bs):
            graphs.append(Data(x=x_vision[i,:,:], edge_index=vision_edge_index))

        data_batch = Batch.from_data_list(graphs) 
        data_batch.edge_index = dropout_edge(data_batch.edge_index, p=self.dropout_edge_ratio, training=self.training)[0]
        return data_batch

    def forward(self, hidden_states:torch.Tensor, pooled_output:torch.Tensor, mode='text'):
        if mode == 'text':
            hidden_states = hidden_states[(len(hidden_states) - self.num_text_layers):] 
            output = torch.cat(hidden_states, dim =-2)
        else:
            hidden_states = hidden_states[(len(hidden_states) - self.num_vision_layers):] 
            output = torch.cat(hidden_states, dim =-2)
        bs = output.shape[0] 
        output = torch.cat([pooled_output.view(bs, 1, -1), output], dim=-2)
        graph = self.build_graph(hidden_states=hidden_states, x=output) 
        graph = self.add_root(graph=graph, bs=bs) 
        graph_output, graph_mean  = self.gnn(graph, batch_size=bs, use_root=(self.root is not None))
        output = self.manifold.get_space(graph_output) + self.manifold.get_space(pooled_output)
        output = self.manifold.add_time(output)
        # output = self.manifold.pt_addition(pooled_output, graph_output)


        self.manifold.assert_check_point_on_manifold(output)
        return output, graph_mean
    

class LorentzGraphModel(nn.Module): 
    def __init__(self, manifold:CustomLorentz , vision_ft_in, text_ft_in, ft_out, config , body, head, manifold_mapper=None, num_layers=1, hidden_size=512, num_hidden_layers=2, shared_proj_layers=False, graph_hidden_channels=512, use_root=False) -> None:
        super().__init__()
        self.config = config
        self.body = body
        self.head = head 
        self.manifold = manifold
        hidden_sizes = [hidden_size] * num_hidden_layers + [ft_out] 
        self.manifold_mapper = manifold_mapper
        self.graph_head = LorentzGraphHead(
            manifold=manifold,
            vision_sizes=[vision_ft_in] * num_layers, 
            text_sizes=[text_ft_in] * num_layers, 
            proj_hidden_sizes=hidden_sizes, 
            ft_out=ft_out,
            graphs_hidden_channel=graph_hidden_channels,
            dropout_edge_ratio=0.5,
            dropout=0.3,
            shared=shared_proj_layers,
            use_root=use_root
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
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
            for hidden_state in outputs.hidden_states:
                lorentz_hidden_states.append(self.manifold_mapper(hidden_state))
            output, graph_output = self.graph_head(hidden_state=lorentz_hidden_states, pooled_output=pooled_output, mode='vision')
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
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)
            for hidden_state in outputs.hidden_states:
                lorentz_hidden_states.append(self.manifold_mapper(hidden_state))

            output, graph_output = self.graph_head(hidden_state=lorentz_hidden_states, pooled_output=pooled_output, mode='text')

        return last_hidden_state, output, graph_output

