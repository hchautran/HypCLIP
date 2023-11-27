
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from .seq_linear import LorentzSeqLinear 
from .graphs import GraphHead, LorentzGraphHead
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from .seq_linear import LorentzSeqLinear
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from torch_geometric.utils import dropout_edge 
from .graphs import LorentzProjLayers, LorentzGNN
from lavis import Blip2Qformer
import time

class Text(object):
    pass

class Blip2Encoder(nn.Module): 
    def __init__(self, config, model:Blip2Qformer, mapper=None ) -> None:
        super().__init__()
        self.config = config
        self.mapper = mapper
        self.ln_vision = model.ln_vision
        self.visual_encoder = model.visual_encoder
        self.query_tokens = model.query_tokens
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.Qformer = model.Qformer
        self.itm_head = model.itm_head

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        vit_embeds = None
        if pixel_values is not None:
            with torch.no_grad():
                vit_embeds = self.ln_vision(self.visual_encoder(pixel_values))
            image_atts = torch.ones(vit_embeds.size()[:-1], dtype=torch.long).to(
                pixel_values.device
            )
            query_tokens = self.query_tokens.expand(vit_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=vit_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            pooled_output = self.vision_proj(query_output.last_hidden_state)
            if self.mapper is not None:
                pooled_output = self.mapper(pooled_output, use_normalized=False)

        else: 
            text_output = self.Qformer.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled_output = self.text_proj(text_output.last_hidden_state[:, 0, :])
            if self.mapper is not None:
                pooled_output = self.mapper(pooled_output, use_normalized=False)


        return vit_embeds, pooled_output

        
