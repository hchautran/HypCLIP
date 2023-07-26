import torch
import torch.nn as nn
from .text_model import CLIPText
from .vision_model import CLIPVision 
from .manifolds.euclidean import Euclidean 
from .manifolds.hyperboloid import Hyperboloid 
from .manifolds.lorentz import Lorentz 
from .manifolds.poincare import PoincareBall 
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput




class HypCLIP(nn.Module):
    def __init__(self, config, logit_scale_init_value=2.6592, curv_learnable=False) -> None:
        super().__init__()
        

        self.model_ckt = config.model_ckt
        self.ft_out = config.ft_out
        self.clip_r = config.clip_radius

        self.text_body = CLIPTextModel.from_pretrained(self.model_ckt) 
        self.vision_body = CLIPVisionModel.from_pretrained(self.model_ckt) 
        self.text_head = nn.Linear(self.text_body.config.hidden_size, self.ft_out, bias=False)
        self.vision_head = nn.Linear(self.vision_body.config.hidden_size, self.ft_out, bias=False)
        self.vision_model = CLIPVision(body=self.vision_body, head=self.vision_head, num_trainable_blocks=config.vision_trainable_blocks)
        self.text_model = CLIPText(body=self.text_body, head=self.text_head, num_trainable_blocks=config.text_trainable_blocks)

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.manifold = config.manifold

        
        curv_learnable = config.curv_learnable
        assert self.manifold in ['poincare', 'hyperboloid', 'lorentz', 'euclidean']

        self.curv = torch.as_tensor(config.curvature if self.manifold != 'euclidean' else 0)
        if not torch.is_floating_point(self.curv):
            self.curv = curv.to(torch.get_default_dtype())
    

        if self.manifold == 'euclidean':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.manifold = Euclidean()
        elif self.manifold == 'poincare':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=curv_learnable)
            self.manifold = PoincareBall()
        elif self.manifold  == 'hyperboloid':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=curv_learnable)
            self.manifold = Hyperboloid()
        else: 
            self.manifold = Lorentz(k=self.curv, learnable=curv_learnable )

    def num_parameters(self, only_trainable=True):
        num_params = 0
        num_params += self.vision_body.num_parameters(only_trainable=only_trainable)
        num_params += self.text_body.num_parameters(only_trainable=only_trainable)

        if only_trainable:
            num_params += sum(p.numel() for p in self.text_head.parameters() if p.requires_grad)
            num_params += sum(p.numel() for p in self.vision_head.parameters() if p.requires_grad)
            num_params += int(self.curv.requires_grad)
        else:
            num_params += sum(p.numel() for p in self.text_head.parameters())
            num_params += sum(p.numel() for p in self.vision_head.parameters())
            num_params += 1 
        return num_params

    def eval(self):
        self.text_body.eval()
        self.vision_head.eval()
        self.text_head.eval()
        self.vision_head.eval()

        
    def dist_func(self, text_embeds, image_embeds):
        if self.curv == 0:
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # cosine similarity as logits
            return torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale
        else:
            text_embeds = self._to_manifold(text_embeds)
            image_embeds = self._to_manifold(image_embeds)
            # square distance on the manifold
            return -self.manifold.sqdist_batch(text_embeds, image_embeds, c=self.curv) * self.logit_scale


        
    def criterion(self, text_embeds , image_embeds):
        bsize = text_embeds.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = self.dist_func(text_embeds, text_embeds) / tau - eye_mask
        logits01 = self.dist_func(text_embeds, image_embeds)/ tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        stats = {
            "logits/min": logits01.min().item(),
            "logits/mean": logits01.mean().item(),
            "logits/max": logits01.max().item(),
            "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        }
        return loss, stats

    def _to_manifold(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
        return self.manifold.proj(self.manifold.expmap0(x, c=self.curv), c=self.curv)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        logit_scale = self.logit_scale.exp()

        sim_per_text = self.dist_func(text_embeds, image_embeds)
        sim_per_image = sim_per_text.t()
        

        loss = None
        if return_loss:
            loss = self.criterion(sim_per_text)

        return dict(
            loss=loss,
            sim_per_image=sim_per_image,
            sim_per_text=sim_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
        
    
if __name__ == '__main__':
    config = {
        'model_ckt': 'openai/clip-vit-base-patch32',
        'ft_out': 512,
        'manifold': 'poincare',
        'curvature': 0.1,
        'clip_radius': 2.3,
    }
    hypCLIP = HypCLIP(config)
    

    print(hypCLIP)

    