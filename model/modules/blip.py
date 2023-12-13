
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers import LorentzLinear
from .seq_linear import  SeqLinear
from transformers import PerceiverConfig
class Text(object):
    pass

class LavisEncoder(nn.Module): 
    def __init__(self, config, vision_body, vision_head, text_body, text_head, mapper=None, use_normalized=False) -> None:
        super().__init__()
        self.vision_body = vision_body
        self.vision_head = vision_head 
        self.text_body = text_body
        self.text_head = text_head 
        self.config = config
        self.mapper = mapper
        self.use_normalized = use_normalized

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is not None:
            # with torch.no_grad():
            outputs = self.vision_body.forward_features(
                pixel_values,
            )
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.vision_head(pooled_output)
            if self.mapper is not None:
                    pooled_output = self.mapper(pooled_output, use_normalized=True)
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.text_body.forward_text(text)

            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.text_head(pooled_output)
            if self.mapper is not None:
                pooled_output = self.mapper(pooled_output, use_normalized=True)


        return last_hidden_state, pooled_output

