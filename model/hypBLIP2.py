from .modules.utils import ManifoldMapper
from model.baseBlip2Model import BaseModel 
from .modules.utils import ManifoldMapper
import torch.nn.functional as F
from .modules.blip2 import Blip2Encoder
import torch


EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"

def get_lora_lavis_blip(config, model):
    return model



class Blip2Model(BaseModel):
    def __init__(self, config, model) -> None:
        super(Blip2Model, self).__init__(config)
        
        mapper = None
        self.config = config
        model = get_lora_lavis_blip(config=config,model=model) 
        self.model = Blip2Encoder(config, model=model, mapper=mapper)
    

    def compute_itm(self, vit_feats, input_ids, attention_mask):
        image_atts = torch.ones(vit_feats.size()[:-1], dtype=torch.long).to(
            vit_feats.device
        )
        query_tokens = self.model.query_tokens.expand(vit_feats.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            vit_feats.device
        )
        attention_mask = torch.cat([query_atts, attention_mask], dim=1)
        output_itm = self.model.Qformer.bert(
            input_ids=input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=vit_feats,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens.size(1), :]
        itm_logit = self.model.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
    
 
    

         