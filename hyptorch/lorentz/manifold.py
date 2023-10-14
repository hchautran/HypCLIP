import torch
from hyptorch.geoopt import Lorentz
from hyptorch import geoopt
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
        return x @ y.T

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

    def sqdist_batch(self, p1_list:torch.Tensor, p2_list:torch.Tensor, dim=-1):
        # device = torch.device("cuda:0" if p1_list.get_device() != -1 else "cpu")
        # dists = torch.tensor([]).to(device)
        # for idx in range(p1_list.shape[0]):
            # cur_dist = self.sqdist(p1_list[idx], p2_list).unsqueeze(0)
        #     dists = torch.cat([dists, cur_dist], dim=0)
        # return dists
        return -2 * self.k - 2 * self.bmm(p1_list, p2_list)
    
    def dist_batch(self, p1_list:torch.Tensor, p2_list:torch.Tensor):
        # device = torch.device("cuda:0" if p1_list.get_device() != -1 else "cpu")
        # dists = torch.tensor([]).to(device)
        # for idx in range(p1_list.shape[0]):
        #     cur_dist = self.dist(p1_list[idx], p2_list).unsqueeze(0)
        #     dists = torch.cat([dists, cur_dist], dim=0)
        d = - self.bmm(p1_list, p2_list)
        return torch.sqrt(self.k) * math.arcosh(d / self.k)
        # return self.dist(p1_list, p2_list)



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
    
    def half_aperture(
        self, x: torch.Tensor, min_radius: float = 0.1, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute the half aperture angle of the entailment cone formed by vectors on
        the hyperboloid. The given vector would meet the apex of this cone, and the
        cone itself extends outwards to infinity.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            min_radius: Radius of a small neighborhood around vertex of the hyperboloid
                where cone aperture is left undefined. Input vectors lying inside this
                neighborhood (having smaller norm) will be projected on the boundary.
            eps: Small float number to avoid numerical instability.

        Returns:
            Tensor of shape `(B, )` giving the half-aperture of entailment cones
            formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
        """

        # Ensure numerical stability in arc-sin by clamping input.
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        asin_input = 2 * min_radius / (torch.norm(x_space, dim=-1) * self.k**0.5 + eps)
        _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

        return _half_aperture


    def oxy_angle(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
        """
        Given two vectors `x` and `y` on the hyperboloid, compute the exterior
        angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
        of the hyperboloid.

        This expression is derived using the Hyperbolic law of cosines.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            y: Tensor of same shape as `x` giving another batch of vectors.

        Returns:
            Tensor of shape `(B, )` giving the required angle. Values of this
            tensor lie in `(0, pi)`.
        """

        # Calculate time components of inputs (multiplied with `sqrt(curv)`):
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        y_space = y.narrow(-1, 1, y.shape[-1] - 1)
        x_time = x.narrow(-1, 0, 1)
        y_time = x.narrow(-1, 0, 1)

        # Calculate lorentzian inner product multiplied with curvature. We do not use
        # the `pairwise_inner` implementation to save some operations (since we only
        # need the diagonal elements).
        c_xyl = self.k * (torch.sum(x_space * y_space, dim=-1) - x_time * y_time)

        # Make the numerator and denominator for input to arc-cosh, shape: (B, )
        acos_numer = y_time + c_xyl * x_time
        acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

        acos_input = acos_numer / (torch.norm(x_space, dim=-1) * acos_denom + eps)
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


        return _angle

    def random(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by uniform distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) 
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)
