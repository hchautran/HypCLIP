import torch
import torch.nn as nn
from .modules.text_model import CLIPText
from .modules.vision_model import CLIPVision 
from .manifolds.euclidean import Euclidean 
from .manifolds.hyperboloid import Hyperboloid 
from .manifolds.lorentz import Lorentz 
from .manifolds.poincare import PoincareBall 
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F




class HypCLIP(nn.Module):
    def __init__(self, config, logit_scale_init_value=2.6592) -> None:
        super().__init__()
        self.config = config
        

        self.model_ckt = config.model_ckt
        self.ft_out = config.ft_out
        self.clip_r = config.clip_radius
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        text_body = CLIPTextModel.from_pretrained(self.model_ckt, cache_dir=config.cache_dir) 
        vision_body = CLIPVisionModel.from_pretrained(self.model_ckt, cache_dir=config.cache_dir) 
        text_head = nn.Linear(text_body.config.hidden_size, self.ft_out, bias=False)
        vision_head = nn.Linear(vision_body.config.hidden_size, self.ft_out, bias=False)

        self.vision_model = CLIPVision(body=vision_body, head=vision_head, num_trainable_blocks=config.vision_trainable_blocks, freeze_embedding=config.freeze_embedding)
        self.text_model = CLIPText(body=text_body, head=text_head, num_trainable_blocks=config.text_trainable_blocks, freeze_embeddings=config.freeze_embedding)

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value, requires_grad=True)
        manifold = config.manifold

        assert manifold in ['poincare', 'hyperboloid', 'lorentz', 'euclidean']

        self.temp = nn.Parameter(torch.as_tensor(config.temp), requires_grad=config.temp != 0) 

        self.curv = torch.as_tensor(config.curv if manifold != 'euclidean' else 0.0)
        if not torch.is_floating_point(self.curv):
            self.curv = self.curv.to(torch.get_default_dtype())
    

        if manifold == 'euclidean':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=False)
            self.manifold = Euclidean()
        elif manifold == 'poincare':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = PoincareBall()
        elif manifold  == 'hyperboloid':
            self.curv = torch.nn.Parameter(self.curv, requires_grad=config.curv_learnable)
            self.manifold = Hyperboloid()
        else: 
            self.manifold = Lorentz(k=self.curv, learnable=config.curv_learnable)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        num_params += self.vision_model.body.num_parameters(only_trainable=only_trainable)
        num_params += self.text_model.body.num_parameters(only_trainable=only_trainable)
        vision_head = self.vision_model.head
        text_head = self.text_model.head
        if only_trainable:
            num_params += sum(p.numel() for p in text_head.parameters() if p.requires_grad)
            num_params += sum(p.numel() for p in vision_head.parameters() if p.requires_grad)
            num_params += int(self.curv.requires_grad)
            num_params += int(self.temp.requires_grad)
            num_params += int(self.logit_scale.requires_grad) 
        else:
            num_params += sum(p.numel() for p in text_head.parameters())
            num_params += sum(p.numel() for p in vision_head.parameters())
            num_params += 3 
        return num_params

    def eval(self):
        self.vision_model.body.eval()
        self.vision_model.head.eval()
        self.text_model.body.eval()
        self.text_model.head.eval()

    def train(self):
        self.vision_model.body.train()
        self.vision_model.head.train()
        self.text_model.body.train()
        self.text_model.head.train()

        
    def dist_func(self, text_embeds, image_embeds):
        logit_scale = self.logit_scale.exp()
        if self.curv == 0:
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # cosine similarity as logits
            return torch.matmul(text_embeds, image_embeds.t()) 
        else:
            # image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = self._to_manifold(text_embeds)
            image_embeds = self._to_manifold(image_embeds)
            # square distance on the manifold
            if self.config.manifold == 'lorentz':
                return -self.manifold.dist_batch(text_embeds, image_embeds)
            return -self.manifold.sqdist_batch(text_embeds, image_embeds, c=self.curv)


        
    def criterion(self, text_embeds , image_embeds):
    
        bsize = text_embeds.shape[0]
        sims_t2i = self.dist_func(text_embeds, image_embeds)/ self.temp 
        target = torch.arange(bsize).to(self.device)
        loss = F.cross_entropy(sims_t2i, target)
        stats = {
            "logits/min": sims_t2i.min().item(),
            "logits/mean": sims_t2i.mean().item(),
            "logits/max": sims_t2i.max().item(),
            "logits/acc": (sims_t2i.argmax(-1) == target).float().mean().item(),
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
        if self.config.manifold == 'lorentz':
            return self.manifold.expmap0(x)
        return self.manifold.proj(self.manifold.expmap0(x, c=self.curv), c=self.curv)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
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
        loss, stats = self.criterion(text_embeds, image_embeds)
        return loss, stats

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_embeds = text_outputs[1]
        return text_embeds


    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        return vision_outputs[1] 


        
    
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

    