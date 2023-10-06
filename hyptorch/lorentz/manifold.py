import torch
from hyptorch.geoopt import Lorentz
from hyptorch.geoopt.manifolds.lorentz import math
from typing import Tuple, Optional


class CustomLorentz(Lorentz):

    def __init__(self, k=1.0, learnable=False, atol=1e-5, rtol=1e-5):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)
        self.atol = atol 
        self.rtol = rtol 


    def sqdist(self, x, y, dim=-1):
        """Squared Lorentzian distance, as defined in the paper 'Lorentzian Distance Learning for Hyperbolic Representation'"""
        return -2 * self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)

    def add_time(self, space):
        """Concatenates time component to given space component."""
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """Calculates time component from given space component."""
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True) ** 2 + self.k)


    def bmm(self, x: torch.Tensor, y: torch.Tensor):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y

    def centroid(self, x, w=None, eps=1e-8):
        """Centroid implementation. Adapted the code from Chen et al. (2022)"""
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=-2)

        denom = -self.inner(avg, avg, keepdim=True)
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def switch_man(self, x, manifold_in: Lorentz):
        """Projection between Lorentz manifolds (e.g. change curvature)"""
        x = manifold_in.logmap0(x)
        return self.expmap0(x)

    def aspt_addition(self, x, y):
        """Parallel transport addition proposed by Chami et al. (2019)"""
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)

    def sqdist_batch(self, p1_list:torch.Tensor, p2_list:torch.Tensor):
        device = torch.device("cuda:0" if p1_list.get_device() != -1 else "cpu")
        dists = torch.tensor([]).to(device)
        for idx in range(p1_list.shape[0]):
            cur_dist = self.sqdist(p1_list[idx], p2_list).unsqueeze(0)
            dists = torch.cat([dists, cur_dist], dim=0)
        return dists
    
    def dist_batch(self, p1_list:torch.Tensor, p2_list:torch.Tensor):
        device = torch.device("cuda:0" if p1_list.get_device() != -1 else "cpu")
        dists = torch.tensor([]).to(device)
        for idx in range(p1_list.shape[0]):
            cur_dist = self.dist(p1_list[idx], p2_list).unsqueeze(0)
            dists = torch.cat([dists, cur_dist], dim=0)
        return dists



    #################################################
    #       Reshaping operations
    #################################################
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements flattening operation directly on the manifold. Based on Lorentz Direct Concatenation (Qu et al., 2022) """
        bs,h,w,c = x.shape
        # bs x H x W x C
        time = x.narrow(-1, 0, 1).view(-1, h*w, 1)
        space = x.narrow(-1, 1, x.shape[-1] - 1).flatten(start_dim=1, end_dim=2) # concatenate all x_s
        time_rescaled = torch.sqrt(torch.sum(time**2, dim=-1, keepdim=True)+(((h*w)-1)/-self.k))
        x = torch.cat([time_rescaled, space], dim=-1)
        return x

    def get_space(self, x):
        return x.narrow(-1, 1, x.shape[-1] - 1)
    
    def get_time(self, x):
        return x.narrow(-1, 0, 1)

    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        """Implements reshaping a flat tensor to an image directly on the manifold. Based on Lorentz Direct Split (Qu et al., 2022)"""
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2] - 1))
        img = self.add_time(space)
        return img

    #################################################
    #       Activation functions
    #################################################
    def lorentz_relu(self, x: torch.Tensor, add_time: bool = True) -> torch.Tensor:
        """Implements ReLU activation directly on the manifold."""
        return self.lorentz_activation(x, torch.relu, add_time)

    def lorentz_activation(
        self, x: torch.Tensor, activation, add_time: bool = True
    ) -> torch.Tensor:
        """Implements activation directly on the manifold."""
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x
    

    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Implements ReLU activation in tangent space."""
        return self.expmap0(torch.relu(self.logmap0(x)))

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, dim=-1, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        dn = x.size(dim) - 1
        x = x**2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(
            dim=dim, keepdim=True
        )
        ok = torch.allclose(quad_form, -self.k, atol=self.atol, rtol=self.rtol)
        if not ok:
            reason = f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        else:
            reason = None
        return ok, reason
