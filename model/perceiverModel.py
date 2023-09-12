import torch
import torch.nn as nn
from .modules.discriminator import Discriminator as DisModel
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval
from transformers import PerceiverConfig
from .modules.utils import freeze_blip 
from .modules.perceiver import MultiModalHead 
from peft import get_peft_model, LoraConfig, TaskType

EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'


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
        self.discriminator = DisModel(self.ft_out, layer_dims=[1])
        model = BlipForImageTextRetrieval.from_pretrained(self.model_ckt)

        head_config = PerceiverConfig(
            d_latents=config.d_latents, 
            num_latents=config.num_latents, 
            num_self_attends_per_block=config.num_self_attends_per_block,
            num_cross_attention_heads=config.num_cross_attention_heads,
            num_self_attention_heads=config.num_self_attention_heads,
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['dense', 'vision_proj', 'text_proj', 'query', 'value','key']
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


        self.vision_body = model.vision_model
        self.text_body =  model.text_encoder
        self.text_head = model.text_proj
        self.vision_head = model.vision_proj
        self.multimodal_head = MultiModalHead(
            head_config, 
            d_vision=model.config.image_text_hidden_size, 
            d_text=model.config.image_text_hidden_size,
            d_out=model.config.image_text_hidden_size,
            num_blocks=self.config.num_blocks
        ) 


    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def eval(self):
        self.vision_body.eval()
        self.vision_body.eval()
        self.multimodal_head.eval()

    def train(self):
        self.vision_body.train()
        self.text_body.train()
        self.multimodal_head.train()
        
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
        if self.config.manifold == EUCLID:
            neg_margin = self.config.euclid_neg_margin * neg_mask 
            pos_margin = self.config.euclid_pos_margin * pos_mask 
            sims_i2t = sims_i2t - neg_margin 
            sims_i2i = sims_i2i - neg_margin 
            sims_i2t = (sims_i2t - pos_margin) * sign 
        else:
            neg_margin = self.config.lorentz_neg_margin * neg_mask 
            pos_margin = self.config.lorentz_pos_margin * pos_mask 
            sims_i2t = sims_i2t + neg_margin 
            sims_i2i = sims_i2i + neg_margin 
            sims_i2t = (sims_i2t + pos_margin) * sign 

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

    def get_attend_mask(self,num_latents):
        zeros = torch.zeros(num_latents)
        hiddens = torch.zeros(num_latents) - float('inf')

        vis_mask = torch.cat([hiddens, zeros]).expand([num_latents, -1])
        text_mask = torch.cat([zeros, hiddens]).expand([num_latents, -1])
        attention_mask = torch.cat([text_mask, vis_mask], dim=0)
        
        return attention_mask.to(self.device)

        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.vision_body(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_body(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        vision_output = self.vision_head(vision_outputs[0])
        text_output = self.text_head(text_outputs[0])
        self_attend_mask = self.get_attend_mask(num_latents=self.config.num_latents)

        text_embeds, image_embeds, text_itm, image_itm = self.multimodal_head(
            text_inputs=text_output, 
            vision_inputs=vision_output, 
            self_attend_mask=self_attend_mask
        ) 
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        

        itm_loss = self.itm_loss(text_itm, image_itm, sims_i2t=sims_i2t)
        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itc_loss + itm_loss
        return loss, stats, itc_loss, itm_loss

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_outputs = self.text_body(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_outputs = self.text_head(text_outputs[0])

        text_embeds = self.multimodal_head.get_text_features(
            text_inputs=text_outputs, 
        ) 

        return text_embeds

    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_body(
            pixel_values=pixel_values,
        )
        vision_outputs = self.vision_head(vision_outputs[0])

        image_embeds = self.multimodal_head.get_vision_features(
            vision_inputs=vision_outputs
        ) 
        
        return image_embeds 

