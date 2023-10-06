import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .seq_linear import SeqLinear
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphConv
import torch.nn.functional as F

class ProjLayers(nn.Module):
  def __init__(self,  sizes=[768], hidden_sizes=[512],  dropout=0.1):
    super().__init__()
    self.projectors = nn.ModuleList([SeqLinear(ft_in=768, layer_dims=hidden_sizes)])
    self.dropout= nn.ModuleList([nn.Dropout(dropout) for _ in sizes])

  def forward(self, hidden_states:torch.Tensor):
    outputs = []
    for i in range(len(self.projectors)):
      output= self.projectors[i](self.dropout[i](hidden_states[i]))
      outputs.append(output)

    return output

class GraphHead(nn.Module):
    def __init__(self,  sizes=[768], proj_hidden_sizes=[512, 512], ft_out=512 ,dropout=0.1):
        super().__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.proj_layers = ProjLayers(sizes, hidden_sizes=proj_hidden_sizes, dropout=dropout)
        self.gnn = GNN(ft_in=proj_hidden_sizes[-1], ft_out=ft_out) 

    def forward(self, hidden_states:torch.Tensor, pooled_output:torch.Tensor):

        ends = []
        starts = []
        begin_index = 1
        hidden_states = hidden_states[len(hidden_states) - self.num_layers :] 
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
        edge_index = torch.tensor([starts, ends], dtype=torch.long)
        graphs = []
        for i in range(bs):
            graphs.append(Data(x=output[i,:,:], edge_index=edge_index))
        return self.gnn(Batch.from_data_list(graphs))


class GNN(torch.nn.Module):
    def __init__(self, ft_in ,hidden_channels, ft_out):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(ft_in, hidden_channels)  
        self.act1 = nn.GELU()
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.act2 = nn.GELU()
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, ft_out)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        return x


