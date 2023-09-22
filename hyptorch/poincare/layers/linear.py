import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch import geoopt
from hyptorch.geoopt import PoincareBall 


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, manifold:PoincareBall,  **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = manifold 
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.reset_parameters()

    def forward(self, input):
        output = self.ball.mobius_matvec(self.weight, input)
        if self.bias is not None:
            output = self.ball.mobius_add(output, self.bias)
        return output

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()

class MobiusAct(nn.Module):
    def __init__(self, *args, manifold:PoincareBall, act,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nonlin = act 
        self.manifold = manifold
   
    def forward(self, x):
        if self.nonlin is not None:
            output = self.manifold.logmap0(x)
            output = self.nonlin(output)
            output = self.manifold.expmap0(output)
        return output 
    
class MobiusLayerNorm(nn.LayerNorm):
    def __init__(self, *args, manifold:PoincareBall, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.weight =  geoopt.ManifoldParameter(self.weight, manifold=self.manifold) 
        self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.manifold)
   
    def forward(self, x):
        if self.nonlin is not None:
            output = self.manifold.logmap0(x)
            output =  F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
            output = self.manifold.expmap0(output)
        return output 
        