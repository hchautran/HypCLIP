import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig
from transformers import Blip2Model


class MultiModalLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, num_self_attend=None) -> None:
        super().__init__()
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
        num_self_attends_per_block = num_self_attend if num_self_attend is not None else config.num_self_attends_per_block
        for _ in range(num_self_attends_per_block):
            self_attention_layers.append(
                PerceiverLayer(
                    config,
                    is_cross_attention=False,
                    num_heads=config.num_self_attention_heads,
                    q_dim=config.d_latents,
                    kv_dim=config.d_latents,
                    widening_factor=config.self_attention_widening_factor,
                )
            )
        self.self_attends=nn.ModuleList(self_attention_layers)

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, text_question, vision_question ,self_attend_mask=None):
        text_output = self.text_layer(
            text_question,
            inputs=text_inputs,
        )

        vision_output = self.vision_layer(
            vision_question,
            inputs=vision_inputs,
        ) 
        text_state = text_output[0] 
        vision_state = vision_output[0] 
        text_itm_state = text_output[0] 
        vision_itm_state = vision_output[0] 

        for layer_module in self.self_attends:
            input = torch.cat([text_state, vision_state], dim=1)
            input_itm = torch.cat([text_itm_state, vision_itm_state], dim=1)
            output = layer_module(input , attention_mask=self_attend_mask)
            itm_state = layer_module(input , attention_mask=None)
            text_state, vision_state = torch.split(output[0], [text_state.shape[1], vision_state.shape[1]] ,dim=1)
            text_itm_state, vision_itm_state = torch.split(itm_state[0], [text_state.shape[1], vision_state.shape[1]] ,dim=1)

        return text_state, vision_state, text_itm_state, vision_itm_state

    def get_vision_features(self, vision_inputs, question):
        vision_output = self.vision_layer(
            question,
            inputs=vision_inputs,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            vision_state = vision_output[0] 
            vision_output = layer_module(vision_state)
        return vision_output[0]

    def get_text_features(self, text_inputs, question, attention_mask=None):
        text_output = self.text_layer(
            question,
            inputs=text_inputs,
            attention_mask=attention_mask,
        ) 
        for _, layer_module in enumerate(self.self_attends):
            text_state = text_output[0]
            text_output = layer_module(text_state)
        return text_output[0]

class MultiModalHead(nn.Module):
    def __init__(self, config:PerceiverConfig, d_vision, d_text, d_out, num_blocks) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.question = nn.Parameter(torch.randn(config.num_latents, config.d_latents))

        multimodal_layers = [MultiModalLayer(config=config, d_vision=d_vision, d_text=d_text)]
        multimodal_layers.extend(
            [MultiModalLayer(config=config, d_text=d_text, d_vision=d_vision) for _ in range(self.num_blocks)]
        )
        self.layers = nn.ModuleList(multimodal_layers)
        self.dropout= nn.Dropout(0.15) 
        self.proj = nn.Linear(config.d_latents + d_out, d_out)


    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, self_attend_mask=None):
        bs = text_inputs.size(0)

        question = self.question.expand([bs, -1, -1])
        text_ori = text_inputs[:, 0, :]
        vision_ori = vision_inputs[:, 0, :]
        itm_vision= []
        itm_text = []
        text_question = question
        vision_question = question

        for i in range(self.num_blocks + 1):
            text_question, vision_question, itm_text_state, itm_vision_state = self.layers[i](
                text_inputs=text_inputs, 
                vision_inputs=vision_inputs, 
                text_question = text_question,
                vision_question = vision_question,
                self_attend_mask=self_attend_mask
            )

        itm_text = itm_text_state[:, 0, :]
        itm_vision = itm_vision_state[:, 0, :]

        text_ouput = text_question[:, 0, :]
        vision_ouput = vision_question[:, 0, :]
        
        text_output = self.proj(self.dropout(torch.cat([text_ori, text_ouput], dim=-1)))
        vision_output = self.proj(self.dropout(torch.cat([vision_ori, vision_ouput], dim=-1)))
        itm_text = self.proj(self.dropout(torch.cat([text_ori, itm_text], dim=-1)))
        itm_vision = self.proj(self.dropout(torch.cat([vision_ori, itm_vision], dim=-1)))

        return text_output, vision_output, itm_text, itm_vision

    def get_vision_features(self, vision_inputs):
        bs = vision_inputs.size(0)
        vision_ori = vision_inputs[:, 0, :]
        vision_question = self.question.expand([bs, -1, -1])

        for i in range(self.num_blocks + 1):
            vision_question = self.layers[i].get_vision_features(vision_inputs, vision_question)
        vision_ouput = vision_question[:, 0, :]

        vision_output = self.proj(self.dropout(torch.cat([vision_ori, vision_ouput], dim=-1)))
        return vision_output

    def get_text_features(self, text_inputs):
        bs = text_inputs.size(0)
        text_ori = text_inputs[:, 0, :]
        text_question = self.question.expand([bs, -1, -1])
        for i in range(self.num_blocks + 1):
            text_question = self.layers[i].get_text_features(text_inputs, text_question)

        text_output = text_question[:, 0, :]
        text_output = self.proj(self.dropout(torch.cat([text_ori, text_output], dim=-1)))
        return text_output





if __name__ == '__main__':
    config = PerceiverConfig(d_latents=512, num_latents=32, num_self_attends_per_block=3)
    model = MultiModalHead(config, d_vision=256, d_text=256,d_out=256 ,num_blocks=6)
    bsize = 100 
    dim = 768
    text = torch.rand(bsize, 30, 256)
    vision = torch.rand(bsize, 344, 256)
    mask = get_attend_mask(bsize, 32)
    print(mask.shape)
    print(mask)
    print(model.num_parameters())
    text_output, vision_output = model(text, vision, mask)
    print(text_output.shape)
    print(vision_output.shape)
    