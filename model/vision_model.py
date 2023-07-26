
import torch
import torch.nn as nn
from .utils import freeze_clip
from typing import Optional


class CLIPVision(nn.Module): 
    def __init__(self, body, head, num_trainable_blocks=0) -> None:
        super().__init__()

        freeze_clip(vision_model=body, num_trainable_blocks=num_trainable_blocks)
        self.body = body
        self.head = head 

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        vision_outputs = self.body(
            pixel_values=pixel_values,
            return_dict=True,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        last_hidden_state = vision_outputs[0]
        projected_image_features = self.head(pooled_output)

        return last_hidden_state, projected_image_features
    

    
    