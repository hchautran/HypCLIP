import torch
import torch.nn as nn
from model.baseModel import BaseModel
from .modules.model import BLIPText
from .modules.vision_model import BLIPVision, CLIPVision
from .modules.swin_model import VisModel 
from transformers import BlipForImageTextRetrieval 
from transformers.models.clip.modeling_clip import CLIPOutput
from .modules.discriminator import Discriminator as DisModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import  Optional, Tuple, Union
from .modules.utils import freeze_blip

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypSwin(BaseModel):
    def __init__(self, config) -> None:
        super(HypSwin, self).__init__(config)
        self.ft_out = config.ft_out

        blip = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
 
        # text_peft_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION, 
        #     inference_mode=False, 
        #     r=8, 
        #     lora_alpha=4, 
        #     lora_dropout=0.1, 
        #     target_modules=['key', 'query']
        # )
    
        # blip = get_peft_model(blip, text_peft_config)
        # print(blip.print_trainable_parameters())

        text_body = blip.text_encoder
        text_head = nn.ModuleList([blip.text_proj])
        freeze_blip(text_model=text_body, text_head=text_head)
        self.discriminator = DisModel(dim=256, layer_dims=[512, 1])
  
        model_ckt = 'microsoft/swinv2-base-patch4-window12-192-22k'

        self.vision_model = VisModel(model_ckt, proj_dim=256, num_stages=2)
        self.text_model = BLIPText(
            config=config,
            body=text_body,
            head=text_head,
        )

    
    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_model(pixel_values=pixel_values),
        return vision_outputs 
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.vision_model(pixel_values=pixel_values) 

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs
        text_embeds = text_outputs[1]
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        itm_loss = self.itm_loss(image_embeds, text_embeds, sims_i2t=sims_i2t)
        stats["logits/itm_loss"] = itm_loss.item() 
        loss = itm_loss + itc_loss
        return loss, stats, itc_loss, itm_loss


