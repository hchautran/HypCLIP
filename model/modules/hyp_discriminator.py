import torch
import torch.nn as nn
import torch.nn.functional as F
from .seq_linear import HypSeqLinear, LorentzSeqLinear
from hyptorch.lorentz.layers import LorentzMLR 
from hyptorch.lorentz.manifold import CustomLorentz 
from hyptorch.poincare.layers import MobiusAct, MobiusLinear 
from hyptorch.poincare.layers.PMLR import UnidirectionalPoincareMLR
class HypDiscriminator(nn.Module):
    def __init__(self, manifold, dim=512, layer_dims=[512], dropout=0.5, act_func='relu', fourier=False):
        super(HypDiscriminator, self).__init__()
        self.manifold = manifold
        self.c = manifold.k 
        self.disc = HypSeqLinear(manifold, ft_in=dim*2, layer_dims=layer_dims, dropout=dropout, act_func=act_func)
        self.mlr = UnidirectionalPoincareMLR(ball=manifold, feat_dim=layer_dims[-1], num_outcome=1) 

    def forward(self, feat1, feat2):
        feat1 = self.manifold.logmap0(feat1)
        feat2 = self.manifold.logmap0(feat2)
        feat = self.manifold.expmap0(torch.cat([feat1, feat2], dim=-1))
        output = self.disc(feat) 
        return self.mlr(output)

class LorentzDiscriminator(nn.Module):
    def __init__(self, manifold:CustomLorentz, dim=512, layer_dims=[512], dropout=0.5, act_func='relu'):
        super(LorentzDiscriminator, self).__init__()
        self.manifold = manifold
        self.disc = LorentzSeqLinear(manifold, ft_in=4*dim+1, layer_dims=layer_dims, dropout=dropout, act_func=act_func)
        self.mlr = LorentzMLR(manifold=manifold, num_features=layer_dims[-1], num_classes=1) 

    def forward(self, feat1, feat2):
        self.manifold.assert_check_point_on_manifold(feat1)
        self.manifold.assert_check_point_on_manifold(feat2)
        feat1_space = self.manifold.get_space(feat1)
        feat2_space = self.manifold.get_space(feat2)

        space = torch.cat([feat1_space, feat2_space, feat1_space - feat2_space, feat1_space + feat2_space], dim=-1)
        out = self.manifold.add_time(space) 
        return self.mlr(self.disc(out))

        
        
