import torch
import torch.nn.functional as F
import torch.nn as nn
from model.manifolds.poincare import PoincareBall
from model.manifolds.nn import HypLinear, HypAct, LorentzLinear
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



def get_activate_func(act_func=None):
    if act_func is None or act_func.lower() == 'id':
        return nn.Identity()
    if act_func.lower() == 'relu':
        return nn.ReLU()
    if act_func.lower() == 'tanh':
        return nn.Tanh()
    if act_func.lower() == 'gelu':
        return nn.GELU()
    if act_func.lower() == 'elu':
        return nn.ELU()


class SeqLinear(nn.Module):
    def __init__(self, ft_in, ft_out=[128], dropout=0.5, act_func='relu'):
        super(SeqLinear, self).__init__()
        self.linear = []
        self.norm = []
        self.dropout = []
        self.act = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(nn.Linear(ft_in, ft_out[idx]))
            else:
                self.linear.append(nn.Linear(ft_out[idx-1], ft_out[idx]))
            if idx != len(ft_out)-1:
                self.norm.append(nn.LayerNorm([ft_out[idx]]))
                self.act.append(get_activate_func(act_func))
            self.dropout.append(nn.Dropout(p=dropout))
            
        self.linear = nn.ModuleList(self.linear)
        for x in self.linear:
            nn.init.kaiming_normal_(x.weight)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        for idx in range(len(self.linear)):
            x = self.dropout[idx](x)
            x = self.linear[idx](x)
            if idx != (len(self.linear)-1): # last layer not use relu
                x = self.act[idx](x)
                x = self.norm[idx](x)
        return x  


class HypSeqLinear(nn.Module):
    def __init__(self, manifold ,ft_in, ft_out, c ,dropout=0.5, act_func='relu'):
        super(HypSeqLinear, self).__init__()
        self.linear = []
        self.norm = []
        self.dropout = []
        self.act = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(HypLinear(manifold, ft_in, ft_out[idx], c))
            else:
                self.linear.append(HypLinear(manifold, ft_out[idx-1], ft_out[idx], c))
            if idx != len(ft_out)-1:
                self.norm.append(nn.LayerNorm([ft_out[idx]]))
                self.act.append(HypAct(manifold, c, c, get_activate_func(act_func)))
            self.dropout.append(nn.Dropout(p=dropout))
            
        self.linear = nn.ModuleList(self.linear)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        for idx in range(len(self.linear)):
            x = self.dropout[idx](x)
            x = self.linear[idx](x)
            if idx != (len(self.linear)-1): # last layer not use relu
                x = self.act[idx](x)
                x = self.norm[idx](x)
        return x  


class LorentzSeqLinear(nn.Module):
    def __init__(self, manifold ,ft_in, ft_out, dropout=0.1, act_func='relu'):
        super(LorentzSeqLinear, self).__init__()
        self.linear = []
        self.norm = []
        self.dropout = []
        self.act = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(LorentzLinear(manifold, ft_in, ft_out[idx], dropout=dropout))
            else:
                self.linear.append(LorentzLinear(manifold, ft_out[idx-1], ft_out[idx], dropout=dropout, nonlin=get_activate_func(act_func)))
            
        self.linear = nn.ModuleList(self.linear)
        
    def forward(self, x):
        for idx in range(len(self.linear)):
            x = self.linear[idx](x)
        return x  

        
        
        
        

class PoincareMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, manifold:PoincareBall, ball_dim, n_classes=2):
        super(PoincareMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.manifold = manifold
        self.reset_parameters()

    def forward(self, x, c):
        p_vals_poincare = self.manifold.expmap0(self.p_vals)

        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = self.manifold.hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))

