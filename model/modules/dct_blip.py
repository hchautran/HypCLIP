
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipPreTrainedModel, 
    BlipConfig, 
    BlipVisionModel, 
    BlipTextModel,
)
from transformers.models.blip.modeling_blip import BlipImageTextMatchingModelOutput
from .dct import dc_transform


class DCTBlipForImageTextRetrieval(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.LongTensor] = None,
        
    ):
        if input_ids is not None:
            return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.get_vision_features(pixel_values=pixel_values)

       
    
    def get_vision_features(self,pixel_values):
        state = self.vision_model.embeddings(pixel_values)
        # state = self.vision_model.pre_layrnorm(state)
        hidden_states = []
        hidden_states.append(state)

        for layer in self.vision_model.encoder.layers:
            state = layer(state, None, None)[0]
            cls = state[:, 0, :].unsqueeze(1)
            state = dc_transform(state[:,1:,:].permute(1,0,2), r=0.8).permute(1,0,2)
            state = torch.cat([cls, state], dim=1)
            # state = dc_transform(state.permute(1,0,2)).permute(1,0,2)
            hidden_states.append(state)

        last_hidden_state = self.vision_model.post_layernorm(state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        vision_embed = self.vision_proj(pooled_output)
        return last_hidden_state, vision_embed

    def get_text_features(self, input_ids, attention_mask):
        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = question_embeds[0] 
        text_embed = self.vision_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed
