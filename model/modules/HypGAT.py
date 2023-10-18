from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt import ManifoldParameter 
from hyptorch.lorentz.layers import LorentzAct, LorentzLinear 
from typing import Any
import torch_sparse
import torch.nn as nn
import math


class LorentzGCN(MessagePassing):
    def __init__(self, manifold:CustomLorentz, ft_in, hidden_channels, dropout=0.4):
        super().__init__(aggr="add")
        self.manifold = manifold
        self.ft_in = ft_in
        self.hidden_channels = hidden_channels
        self.linear = LorentzLinear(manifold, ft_in + 1, hidden_channels + 1, dropout=dropout)
        # self.act = LorentzAct(nn.GELU(), manifold=manifold)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        self.manifold.assert_check_point_on_manifold(x)
        out= self.manifold.projx(out + x)
        self.manifold.assert_check_point_on_manifold(x)
        return self.linear(out)

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        output = adj_t.matmul(x)
        return output 


class LorentzGAT(MessagePassing):
   
    _alpha: OptTensor

    def __init__(
        self,
        manifold:CustomLorentz,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.manifold = manifold
        self.act = LorentzAct(nn.LeakyReLU(negative_slope=negative_slope), manifold=manifold)

        self.lin_l = LorentzLinear(manifold, in_channels + 1, out_channels+1, bias=bias, dropout=dropout) 
        self.lin_r = LorentzLinear(manifold, in_channels + 1, out_channels+1, bias=bias, dropout=dropout)
        self.att = ManifoldParameter(
            data=self.manifold.random((1, self.out_channels + 1)),
            manifold=manifold, 
            requires_grad=True
        )

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, return_attention_weights: bool = None):

        x_l = self.lin_l(x)
        x_r = self.lin_r(x)

        # self.manifold.assert_check_point_on_manifold(x_l)
        # self.manifold.assert_check_point_on_manifold(x_r)
        # print('got here')
        assert x_l is not None
        assert x_r is not None

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.manifold.projx(self.propagate(edge_index, x=(x_l, x_r), size=None))
        # print(out.shape)
        self.manifold.assert_check_point_on_manifold(out)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        return out

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x = self.manifold.projx(x_i + x_j)
        x = self.act(x)
        self.manifold.assert_check_point_on_manifold(x)
        alpha =  -(self.manifold.get_space(x) * self.manifold.get_space(self.att)).sum(dim=-1) + (self.manifold.get_time(x) * self.manifold.get_time(self.att)).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j  * alpha.unsqueeze(-1) 

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

