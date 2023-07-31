import torch
import torch.nn as nn
from .seq_linear import SeqLinear
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, dim=512, ft_out=[512,1], dropout=0.5, act_func='relu'):
        super(Discriminator, self).__init__()
        self.disc = SeqLinear(ft_in=dim*4, ft_out=ft_out, dropout=dropout, act_func=act_func)

    def forward(self, feat1, feat2):
        dist = torch.abs(feat1-feat2)
        mul = torch.mul(feat1, feat2)
        return torch.sigmoid(self.disc(torch.cat([feat1, feat2, dist, mul], dim=1)))

        
class CoDiscriminator(nn.Module):
    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, dim=512, fourier=True):
        super(CoDiscriminator, self).__init__()

        self.embedding_dim = dim 
        self.latent_dim = dim  
        self.k = 256 
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Wi = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.wHi = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))

        #register weights and biAi Ai params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Wi", self.Wi)
        self.register_parameter("wHi", self.wHi)
        self.register_parameter("whc", self.whc)

        #concatenation operation for hyperbolic 
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Wi.data = torch.randn((self.k, self.latent_dim))
        self.wHi.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.fourier = fourier
        self.disc = SeqLinear(ft_in=dim*4, ft_out=[512, 1], dropout=0.5, act_func='relu')

    def forward(self, img_rep, cap_rep):
        if self.fourier:
            img_rep_fourier = torch.fft.fft2(img_rep).float()
            cap_rep_fourier = torch.fft.fft2(cap_rep).float()
            img_rep_trans = img_rep_fourier.transpose(-1, -2)#[bs, dim, len]
            cap_rep_trans = cap_rep_fourier.transpose(-1, -2)#[bs, dim, len]
            L = torch.tanh(torch.matmul(torch.matmul(img_rep_fourier, self.Wl), img_rep_trans))  
            L_trans = L.transpose(-1, -2)
        else:
            img_rep_trans = img_rep.transpose(-1, -2)#[bs, dim, len]
            cap_rep_trans = cap_rep.transpose(-1, -2)#[bs, dim, len]
            L = torch.tanh(torch.matmul(torch.matmul(img_rep, self.Wl), img_rep_trans))  
            L_trans = L.transpose(-1, -2)

        Hi = torch.tanh(torch.matmul(self.Wi, img_rep_trans) + torch.matmul(torch.matmul(self.Wc, cap_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, cap_rep_trans)+ torch.matmul(torch.matmul(self.Wi, img_rep_trans), L_trans))
        Ai = F.softmax(torch.matmul(self.wHi, Hi), dim=-1)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=-1)

        co_s = torch.matmul(Ai,img_rep) # (1, dim)
        co_c = torch.matmul(Ac, cap_rep) # (1, dim)
        co_sc = torch.cat([co_s, co_c], dim = -1)
        co_sc = torch.squeeze(co_sc) # [bs, dim*2], 
        print(co_s.shape)
        print(co_c.shape)
        print(co_sc.shape)
        return torch.sigmoid(self.disc(co_sc))