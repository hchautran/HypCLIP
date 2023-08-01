
import torch
import torch.nn as nn
from .utils import freeze_clip
from typing import Optional
from .seq_linear import LorentzSeqLinear 


class CLIPVision(nn.Module): 
    def __init__(self, config, manifold, body, head, num_trainable_blocks=0, freeze_embedding=True, use_hyp_linear=True) -> None:
        super().__init__()

        freeze_clip(vision_model=body, num_trainable_blocks=num_trainable_blocks, freeze_embeddings=freeze_embedding)
        self.body = body
        self.head = head 
        self.linear = None
        if use_hyp_linear:
            self.linear = LorentzSeqLinear(manifold, ft_in=config.ft_out, ft_out=[config.ft_out]) 

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
        if self.linear is not None:
            projected_image_features = self.linear(projected_image_features)

        return last_hidden_state, projected_image_features
    

    
    