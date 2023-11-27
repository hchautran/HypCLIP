
import torch
import torch.nn as nn
from .utils import freeze_clip, freeze_blip 
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from typing import Optional
from hyptorch.lorentz.manifold import CustomLorentz
from torch_geometric.utils import dropout_edge 
from .perceiver import MultiModalModel 
from hyptorch.lorentz.layers import LorentzMLR 
from .seq_linear import LorentzSeqLinear, SeqLinear
from transformers import PerceiverConfig
from typing import List

class Text(object):
    pass

class BLIPEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:,0,:]
        return last_hidden_state, pooled_output 

class CLIPEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )
        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        pooled_output = outputs[1]
        last_hidden_state = outputs[0]
        return last_hidden_state, pooled_output 

class LavisEncoder(nn.Module): 
    def __init__(self, body) -> None:
        super().__init__()
        self.body= body 

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is not None:
            outputs = self.body.forward_features(
                pixel_values,
            )
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
        else:
            text = Text() 
            text.input_ids=input_ids
            text.attention_mask=attention_mask
            outputs = self.body.forward_text(text)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]

        return last_hidden_state, pooled_output


class FuseEncoder(nn.Module): 
    def __init__(self, config, vision_bodies, text_bodies, vision_head, text_head, mapper=None, use_normalized=False ) -> None:
        super().__init__()
        self.vision_bodies = nn.ModuleList([vision_bodies]) 
        self.text_bodies = nn.ModuleList([text_bodies]) 
        self.vision_head = vision_head 
        self.text_head = text_head
        self.config = config
        self.mapper = mapper
        self.use_normalized = use_normalized

    def forward(
            self,
            pixel_values: List[torch.FloatTensor] = None,
            input_ids: List[torch.Tensor] = None,
            attention_mask: List[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        last_hidden_states = []
        if pixel_values is not None:
            for i in range(len(pixel_values)):
                with torch.no_grad:
                    last_hidden_state, pooled_output = self.vision_bodies[i](
                        pixel_values=pixel_values[i]
                    )
            last_hidden_states.append(last_hidden_state)
        else:
            for i in range(len(input_ids)):
                with torch.no_grad:
                    last_hidden_state, pooled_output = self.text_bodies[i](
                        input_ids=input_ids[i], 
                        attention_mask=attention_mask[i]
                    )
            last_hidden_states.append(last_hidden_state)

        if self.mapper is not None:
            pooled_output = self.mapper(pooled_output, use_normalized=False)


        return last_hidden_state, pooled_output
    
    def get_text_features(self, input_ids:List[torch.Tensor], attention_mask:List[torch.Tensor]):
        pass

    def get_vision_features(self, pixel_values:List[torch.Tensor]):
        pass

    def compute_itm(self, vision_hidden_states:torch.Tensor, text_hidden_states:torch.Tensor): 
        itm, _ = self.perceiver_head(
            text_inputs=text_hidden_states, 
            vision_inputs=vision_hidden_states, 
            self_attend_mask=None,
        ) 

        # itm_vision = self.manifold_mapper(itm_vision)
        itm = self.manifold_mapper(itm)
        # hidden_states = torch.cat([itm_vision, itm_text], dim=1)
        itm = self.perceiver_proj(itm) 
        itm_score = self.itm_head(itm)
        return itm_score 







 

