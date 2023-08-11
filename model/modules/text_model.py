
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from .seq_linear import LorentzSeqLinear 



class CLIPText(nn.Module): 
    def __init__(self, body, head, num_trainable_blocks=0, freeze_embeddings=True) -> None:
        super().__init__()

        freeze_clip(text_model=body, num_trainable_blocks=num_trainable_blocks, freeze_embeddings=freeze_embeddings)
        self.body = body
        self.head = head 
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        last_hidden_state = text_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        for layer in self.head: 
            pooled_output = layer(pooled_output)

        return last_hidden_state, pooled_output
    

        
class BLIPText(nn.Module): 
    def __init__(self, body, head, num_trainable_blocks=0, freeze_embeddings=True) -> None:
        super().__init__()

        freeze_blip(text_model=body, num_trainable_blocks=num_trainable_blocks, freeze_embeddings=freeze_embeddings)
        self.body = body
        self.head = head 
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        text_outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        last_hidden_state = text_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        for layer in self.head: 
            pooled_output = layer(pooled_output)

        return last_hidden_state, pooled_output
    

    
