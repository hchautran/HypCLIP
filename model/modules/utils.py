from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import BlipVisionModel, BlipTextModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz


def fr(m):
    for param in m.parameters():
        param.requires_grad = False


def freeze_clip(
    vision_model: CLIPVisionModelWithProjection = None,
    text_model: CLIPTextModelWithProjection = None,
    freeze_embeddings=True,
    num_trainable_blocks=-1,
):
    if num_trainable_blocks == -1:
        return

    if vision_model is not None:
        if freeze_embeddings:
            fr(vision_model.encoder.embeddings)
        for idx in range(len(vision_model.encoder.layers) - num_trainable_blocks):
            fr(vision_model.encoder.layers[idx])

    if text_model is not None:
        if freeze_embeddings:
            fr(text_model.encoder.embeddings)
        for idx in range(len(text_model.encoder.layers) - num_trainable_blocks):
            fr(text_model.encoder.layers[idx])


def freeze_blip(
    vision_model: BlipVisionModel = None,
    text_model: BlipTextModel = None,
    freeze_embeddings=True,
    num_trainable_blocks=-1,
):
    if num_trainable_blocks == -1:
        return

    if vision_model is not None:
        if freeze_embeddings:
            fr(vision_model.embeddings)
        for idx in range(len(vision_model.encoder.layers) - num_trainable_blocks):
            fr(vision_model.encoder.layers[idx])

    if text_model is not None:
        if freeze_embeddings:
            fr(text_model.embeddings)
        for idx in range(len(text_model.encoder.layer) - num_trainable_blocks):
            fr(text_model.encoder.layer[idx])


class ManifoldMapper(nn.Module):
    def __init__(self, manifold:CustomLorentz, curv, clip_r=None):
        super().__init__()
        self.manifold = manifold
        self.curv = curv
        self.clip_r = clip_r

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
            x = x * fac
        
        x = F.pad(x, (1,0), "constant", 0)
        
        out = self.manifold.expmap0(x)
        return out 


class LorentzCentroidPooler(nn.Module):
    def __init__(self, manifold: CustomLorentz, curv, clip_r=None):
        super().__init__()
        self.manifold = manifold
        self.curv = curv
        self.clip_r = clip_r

    def forward(self, x, attention_mask=None):
        pooled_x = self.manifold.centroid(x, attention_mask=attention_mask)
        return pooled_x
