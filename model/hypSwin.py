import torch
import torch.nn as nn
from model.baseModel import BaseModel
from .modules.text_model import CLIPText
from .modules.swin_model import VisModel 
from transformers import CLIPTextModelWithProjection
from transformers.models.clip.modeling_clip import CLIPOutput
from peft import get_peft_model, LoraConfig, TaskType
from typing import  Optional, Tuple, Union

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypSwin(BaseModel):
    def __init__(self, config) -> None:
        super(HypSwin, self).__init__(config)

        text_model = CLIPTextModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
 
        text_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['k_proj', 'v_proj', 'q_proj']
        )
    
        text_model = get_peft_model(text_model, text_peft_config)
        print(text_model.print_trainable_parameters())

        text_body = text_model.text_model
        text_head = nn.ModuleList([text_model.text_projection])
        model_ckt = 'microsoft/swinv2-base-patch4-window12-192-22k'

        self.vision_model = VisModel(model_ckt, proj_dim=256, num_stages=2)
        self.text_model = CLIPText(
            config=config,
            body=text_body,
            head=text_head,
            num_trainable_blocks=config.text_trainable_blocks,
            freeze_embeddings=config.freeze_embedding,
        )

    
    def get_vision_features(self, pixel_values:torch.Tensor):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        return vision_outputs 
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

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


