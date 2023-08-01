
import torch
import torch.nn as nn
from .utils import freeze_clip
from typing import Optional
from .seq_linear import LorentzSeqLinear, HypSeqLinear


class CLIPText(nn.Module): 
    def __init__(self, config ,manifold ,body, head, num_trainable_blocks=0, freeze_embeddings=True, use_hyp_linear=True) -> None:
        super().__init__()

        freeze_clip(text_model=body, num_trainable_blocks=num_trainable_blocks, freeze_embeddings=freeze_embeddings)
        self.body = body
        self.head = head 
        self.use_hyp_linear=use_hyp_linear
        self.linear = None
        if use_hyp_linear:
            self.linear = LorentzSeqLinear(manifold,ft_in=config.ft_out, ft_out=[config.ft_out]) 
        
        

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

        pooled_output = text_outputs[1]
        last_hidden_state = text_outputs[0]

        projected_text_features = self.head(pooled_output)
        if self.linear is not None:
            projected_text_features = self.linear(projected_text_features)

        return last_hidden_state, projected_text_features
    

        

    

    
    