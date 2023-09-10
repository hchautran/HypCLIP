import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig


class MultiModalLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text) -> None:
        super().__init__()
        self.question= nn.Parameter(torch.randn(config.num_latents, config.d_latents))
        self.vision_layer= PerceiverLayer(
            config,
            kv_dim=d_vision,
            q_dim=config.d_latents,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self.text_layer= PerceiverLayer(
            config,
            kv_dim=d_text,
            q_dim=config.d_latents,
            is_cross_attention=True,
            use_query_residual=True,
            num_heads=config.num_cross_attention_heads,
            widening_factor=config.cross_attention_widening_factor,
        )
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = PerceiverLayer(
                config,
                is_cross_attention=False,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)
        self.self_attends = nn.ModuleList(self_attention_layers)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, attention_mask=None):
        question = self.question.expand(text_inputs.size(0), -1, -1)

        text_output = self.text_layer(
            question,
            inputs=text_inputs,
            # inputs_mask=attention_mask,
        )
        vision_output = self.vision_layer(
            question,
            inputs=vision_inputs,
        ) 

        for _, layer_module in enumerate(self.self_attends):
            text_state = text_output[0] 
            vision_state = vision_output[0] 
            text_output = layer_module(text_state)
            vision_output = layer_module(vision_state)

        return text_output[0], vision_output[0]

    def get_vision_features(self, vision_inputs):
        question = self.question.expand(vision_inputs.size(0), -1, -1)
        vision_output = self.vision_layer(
            question,
            inputs=vision_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            vision_state = vision_output[0] 
            vision_output = layer_module(vision_state)
        return vision_output

    def get_text_features(self, text_inputs, attention_mask=None):
        question = self.question.expand(text_inputs.size(0), -1, -1)
        text_output = self.text_layer(
            question,
            inputs=text_inputs,
            attention_mask=attention_mask,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            text_state = text_output[0]
            text_output = layer_module(text_state)
        return text_output


if __name__ == '__main__':
    config = PerceiverConfig(d_latents=1024, num_latents=128, num_self_attends_per_block=1)
    model = MultiModalLayer(config, d_vision=768, d_text=768)
    bsize = 100 
    dim = 768
    text = torch.rand(bsize, 30, 768)
    vision = torch.rand(bsize, 224, 768)
    print(model)
    print(model.num_parameters())
    # text_output, vision_output = model(text, vision)
    text_output = model.get_text_features(text)
    vision_output = model.get_vision_features(vision)
    print(text_output[0].shape)
    print(vision_output[0].shape)
    