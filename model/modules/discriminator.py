import torch
import torch.nn as nn
from .seq_linear import SeqLinear
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, dim=512, layer_dims=[512,1], dropout=0.5, act_func='relu'):
        super(Discriminator, self).__init__()
        self.disc = SeqLinear(ft_in=dim*4 , layer_dims=layer_dims, dropout=dropout, act_func=act_func)

    def forward(self, feat1, feat2):
        return torch.sigmoid(self.disc(torch.cat([feat1, feat2, feat1-feat2, feat1+feat2], dim=1)))

        
