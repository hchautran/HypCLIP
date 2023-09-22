import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from .modules.utils import freeze_blip 
from .modules.perceiver import CoHead 
from peft import get_peft_model, LoraConfig, TaskType
from typing import List 
from .modules.seq_linear import SeqLinear

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
BLIP_BASE_FLICKR = "Salesforce/blip-itm-base-flickr"
BLIP_LARGE_FLICKR = "Salesforce/blip-itm-large-flickr"
BLIP_BASE_COCO = "Salesforce/blip-itm-base-coco"
BLIP_LARGE_COCO = "Salesforce/blip-itm-large-coco"
CLIP_BASE_PATCH_32 = "openai/clip-vit-base-patch32"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"
CLIP_LARGE_PATCH_14 = "openai/clip-vit-large-patch14"


class ConvPooler(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.pooler = nn.Conv1d(
            in_channels=config.d_latents, 
            out_channels=config.ft_out,
            kernel_size=config.num_latents,
            stride=config.num_latents,
        )
        
    def forward(self, input:torch.Tensor):
        input = input.transpose(-1, -2)
        output = self.pooler(input)
        return output.squeeze(-1)
        
        

class MyModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_ckt = config.model_ckt
        self.ft_out = config.ft_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weight_i2t = config.weight_i2t
        self.temp = nn.Parameter(torch.as_tensor(config.temp), requires_grad=config.temp != 0) 
        self.discriminator = DisModel(dim=256, layer_dims=[512, 1])
        blip_model = BlipForImageTextRetrieval.from_pretrained(BLIP_BASE_FLICKR)
        clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_BASE_PATCH_16)
        clip_text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_BASE_PATCH_16)


        # self.text_co_head = CoHead(embedding_dim_1=256, embedding_dim_2=512, dim=256, ft_out=256, fourier=False) 
        # self.vision_co_head = CoHead(embedding_dim_1=256, embedding_dim_2=512, dim=256, ft_out=256, fourier=False) 
        self.text_co_head= SeqLinear(ft_in=(256 + 512) , layer_dims=[512, 256], dropout=0.2, act_func='gelu')
        self.vision_co_head= SeqLinear(ft_in=(256 + 512) , layer_dims=[512, 256], dropout=0.2, act_func='gelu')

        vision_target_modules = ['visual_projection']
        for i in range(config.vision_trainable_blocks): 
            index = 11 - i
            vision_target_modules.extend([
                # 'patch_embedding',
                # f'*{index}.mlp.fc1', 
                # f'*{index}.mlp.fc2', 
                f'*{index}.self_attn.out_proj',
                f'*{index}.self_attn.q_proj',
                f'*{index}.self_attn.k_proj',
                f'*{index}.self_attn.v_proj', 
            ])

 
        blip_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=8, 
            lora_dropout=0.1, 
            target_modules=['vision_proj', 'text_proj']
        )
        text_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=[
                'k_proj', 
                'v_proj', 
                'q_proj', 
                'out_proj', 
                # 'fc1', 
                # 'fc2', 
                'text_projection'
            ]
        )
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.1, 
            target_modules=vision_target_modules
        )

        blip_model = get_peft_model(blip_model, blip_peft_config)
        clip_vision_model = get_peft_model(clip_vision_model, vision_peft_config)
        clip_text_model = get_peft_model(clip_text_model, text_peft_config)


        self.vision_bodies = nn.ModuleList([blip_model.vision_model, clip_vision_model.vision_model])
        self.text_bodies =  nn.ModuleList([blip_model.text_encoder, clip_text_model.text_model])
        self.vision_heads = nn.ModuleList([blip_model.vision_proj, clip_vision_model.visual_projection])
        self.text_heads = nn.ModuleList([blip_model.text_proj, clip_text_model.text_projection ])

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def eval(self):
        self.text_bodies.eval()
        self.vision_bodies.eval()
        self.text_heads.eval()
        self.vision_heads.eval()
        self.text_co_head.eval()
        self.vision_co_head.eval()
        self.discriminator.eval()

    def train(self):
        self.vision_bodies.train()
        self.text_bodies.train()
        self.vision_heads.train()
        self.text_heads.train()
        self.text_co_head.train()
        self.vision_co_head.train()
        self.discriminator.train()
        
    def dist_func(self, x, y):
        x = F.normalize(x,p=2, dim=-1) 
        y = F.normalize(y,p=2, dim=-1) 
        return torch.matmul(x, y.t()) 


    def itm_loss(self, imgs, cap, sims_i2t):
        bs = imgs.shape[0]
        weights_i2t = F.softmax(sims_i2t, dim=1)
        weights_t2i = F.softmax(sims_i2t.T, dim=1)
        mask = (torch.eye(bs) > 0).to(self.device)

        weights_i2t.masked_fill_(mask, 0)
        weights_t2i.masked_fill_(mask, 0) 
        # select a negative image for each text
        img_enc_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            img_enc_neg.append(imgs[neg_idx])
        img_enc_neg = torch.stack(img_enc_neg,dim=0) 

        # select a negative text for each image
        cap_enc_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            cap_enc_neg.append(cap[neg_idx])
        cap_enc_neg = torch.stack(cap_enc_neg,dim=0)   

        cap_enc_all = torch.cat([cap, cap, cap_enc_neg],dim=0)     
        img_enc_all = torch.cat([imgs, img_enc_neg, imgs],dim=0)
        itm_labels = torch.cat(
            [
                torch.ones(bs,dtype=torch.float),
                torch.zeros(2*bs,dtype=torch.float)
            ],
        dim=0).view(-1,1).to(imgs.device)

        disc = self.discriminator(img_enc_all, cap_enc_all)
        loss_itm = F.binary_cross_entropy_with_logits(disc, itm_labels)
        return loss_itm


    def margin_loss(self, sims_i2t, sims_i2i):
        bsize = sims_i2t.shape[0] 
        ones = torch.ones(bsize, bsize).to(self.device)
        pos_mask = torch.eye(bsize).to(self.device) 
        neg_mask = torch.ne(ones, pos_mask).float().to(self.device)
        sign = ones.masked_fill_(torch.eq(ones, pos_mask), -1.0) 
        neg_margin = self.config.euclid_img_neg_margin * neg_mask 
        pos_margin = self.config.euclid_pos_margin * pos_mask 
        sims_i2t = sims_i2t - neg_margin 
        sims_i2i = sims_i2i - neg_margin 
        sims_i2t = (sims_i2t - pos_margin) * sign 
     
        sims = torch.cat([torch.clamp(sims_i2t, min=0.0) , torch.clamp(sims_i2i, min=0.0)], dim=-1) 
        loss =  torch.mean(torch.sum(sims.pow(2),dim=-1), dim=0) 
        return loss
        
    def itc_loss(self, image_embeds , text_embeds):
        bsize = text_embeds.shape[0]
        eye_mask = torch.eye(bsize).to(self.device) * 1e9
        sims_i2t = self.dist_func(image_embeds, text_embeds)
        sims_t2i = sims_i2t.T
        sims_i2i = self.dist_func(image_embeds, image_embeds)
        sims_t2t = self.dist_func(text_embeds, text_embeds)
        target = torch.arange(bsize).to(self.device)
        logits_i2t = torch.cat([sims_i2t/self.temp, sims_t2t/self.temp - eye_mask], dim=1)
        logits_t2i = torch.cat([sims_t2i/self.temp, sims_i2i/self.temp - eye_mask], dim=1)

        margin_loss = self.weight_i2t * self.margin_loss(sims_i2t, sims_t2t - eye_mask) + (1 - self.weight_i2t) * self.margin_loss(sims_t2i, sims_i2i - eye_mask)
        itc_loss =  self.weight_i2t * F.cross_entropy(logits_i2t, target) + (1 - self.weight_i2t) * F.cross_entropy(logits_t2i, target) 
        loss = itc_loss + margin_loss 
        
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/margin_loss": margin_loss.item() if margin_loss is not None else 0.0,
            "logits/itc_loss": itc_loss.item(),
            "logits/min": sims_i2t.min().item(),
            "logits/mean": sims_i2t.mean().item(),
            "logits/max": sims_i2t.max().item(),
            "logits/acc": (sims_i2t.argmax(-1) == target).float().mean().item(),
        }
        return loss, stats, sims_i2t

    def forward(
        self,
        blip_input_ids: Optional[torch.LongTensor] = None,
        blip_pixel_values: Optional[torch.FloatTensor] = None,
        blip_attention_mask: Optional[torch.Tensor] = None,
        clip_input_ids: Optional[torch.LongTensor] = None,
        clip_pixel_values: Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CLIPOutput]:
                
        blip_vision_outputs = self.vision_bodies[0](
            pixel_values=blip_pixel_values,
        )
        blip_vision_ori = self.vision_heads[0](blip_vision_outputs[1])

        clip_vision_outputs = self.vision_bodies[1](
            pixel_values=clip_pixel_values,
        )
        clip_vision_ori = self.vision_heads[1](clip_vision_outputs[1])

        image_embeds = self.vision_co_head(torch.concat([blip_vision_ori, clip_vision_ori], dim=-1)) 


        blip_text_outputs = self.text_bodies[0](
            input_ids=blip_input_ids,
            attention_mask=blip_attention_mask,
            return_dict=True
        )
        blip_text_ori = self.text_heads[0](blip_text_outputs[0][:,0,:])

        clip_text_outputs = self.text_bodies[1](
            input_ids=clip_input_ids,
            attention_mask=clip_attention_mask,
            return_dict=True,
        )
        clip_text_ori = self.text_heads[1](clip_text_outputs[1])

        text_embeds = self.text_co_head(torch.concat([blip_text_ori, clip_text_ori], dim=-1) ) 
        
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        itm_loss = self.itm_loss(text_embeds, image_embeds, sims_i2t=sims_i2t)

        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itc_loss + itm_loss
        return loss, stats, itc_loss, itm_loss

    def get_text_features(
        self,
        blip_input_ids: torch.Tensor, 
        blip_attention_mask: torch.Tensor,
        clip_input_ids: torch.Tensor, 
        clip_attention_mask: torch.Tensor,
    ):
        blip_text_outputs = self.text_bodies[0](
            input_ids=blip_input_ids,
            attention_mask=blip_attention_mask,
        )
        blip_text_ori = self.text_heads[0](blip_text_outputs[0][:,0,:])

        clip_text_outputs = self.text_bodies[1](
            input_ids=clip_input_ids,
            attention_mask=clip_attention_mask,
        )
        clip_text_ori = self.text_heads[1](clip_text_outputs[1])
        

        text_embeds = self.text_co_head(torch.concat([blip_text_ori, clip_text_ori], dim=-1) ) 
        
        return text_embeds 


    def get_vision_features(self, blip_pixel_values, clip_pixel_values):
            
        blip_vision_outputs = self.vision_bodies[0](
            pixel_values=blip_pixel_values,
        )
        blip_vision_ori = self.vision_heads[0](blip_vision_outputs[1])
        # blip_vision_ori =  blip_vision_output[:, 0, :] 

        clip_vision_outputs = self.vision_bodies[1](
            pixel_values=clip_pixel_values,
        )
        clip_vision_ori = self.vision_heads[1](clip_vision_outputs[1])
        # clip_vision_ori = clip_vision_output[:, 0, :] 
        

        image_embeds = self.vision_co_head(torch.concat([blip_vision_ori, clip_vision_ori], dim=-1)) 
        
        return image_embeds 

