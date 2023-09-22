import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import PerceiverAttention, PerceiverMLP
from transformers import PerceiverLayer
from transformers import PerceiverConfig
import torch.nn.functional as F

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
            itm_state = layer_module(input_itm, attention_mask=None)[0]

            itm_state = itm_state + output[0]
            text_state, vision_state = torch.split(output[0], [text_state.shape[1], vision_state.shape[1]] ,dim=1)
            text_itm_state, vision_itm_state = torch.split(itm_state, [text_state.shape[1], vision_state.shape[1]] ,dim=1)

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
        self.text_question = nn.Parameter(torch.empty(config.num_latents, config.d_latents))
        self.vision_question= nn.Parameter(torch.empty(config.num_latents, config.d_latents))
        nn.init.kaiming_normal_(self.text_question, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.vision_question, mode='fan_out', nonlinearity='leaky_relu')

        multimodal_layers = [MultiModalLayer(config=config, d_text=d_text, d_vision=d_vision) for _ in range(self.num_blocks)]
        self.layers = nn.ModuleList(multimodal_layers)
        self.dropout= nn.Dropout(0.1) 
        self.proj = nn.Linear(config.d_latents , d_out)


    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_ori, vision_ori ,text_inputs, vision_inputs, self_attend_mask=None):
        bs = text_inputs.size(0)

        itm_vision= []
        itm_text = []
        text_question = self.text_question.expand([bs, -1, -1])
        vision_question = self.vision_question.expand([bs, -1 ,-1]) 
        res_text = text_question
        res_vision = vision_question

        for i in range(self.num_blocks):
            text_question, vision_question, itm_text_state, itm_vision_state = self.layers[i](
                text_inputs=text_inputs, 
                vision_inputs=vision_inputs, 
                text_question = text_question,
                vision_question = vision_question,
                self_attend_mask=self_attend_mask
            )
            text_question = text_question + res_vision
            vision_question = vision_question + res_text

        itm_text = torch.mean(itm_text_state, dim=1)
        itm_vision = torch.mean(itm_vision_state, dim=1)

        text_output = torch.mean(text_question, dim=1) 
        vision_output = torch.mean(vision_question, dim=1)
        
        text_output = self.proj(self.dropout(text_output)) + text_ori
        vision_output = self.proj(self.dropout(vision_output)) + vision_ori
        itm_text = self.proj(self.dropout(itm_text)) + text_ori
        itm_vision = self.proj(self.dropout(itm_vision)) + vision_ori

        return text_output, vision_output, itm_text, itm_vision

    def get_vision_features(self, vision_ori, vision_inputs):
        bs = vision_inputs.size(0)
        vision_question = self.vision_question.expand([bs, -1, -1])
        text_question = self.text_question.expand([bs, -1, -1])
        res_text = text_question 

        for i in range(self.num_blocks):
            vision_question = self.layers[i].get_vision_features(vision_inputs, vision_question)
            vision_question = vision_question + res_text

        vision_ouput = torch.mean(vision_question, dim = 1)
        vision_output = self.proj(self.dropout(vision_ouput)) + vision_ori
        return vision_output

    def get_text_features(self, text_ori, text_inputs):
        bs = text_inputs.size(0)
        text_question = self.text_question.expand([bs, -1, -1])
        vision_question = self.vision_question.expand([bs, -1, -1])
        res_vision = vision_question
        for i in range(self.num_blocks):
            text_question = self.layers[i].get_text_features(text_inputs, text_question)
            text_question = text_question + res_vision 

        text_output = torch.mean(text_question, dim = 1)

        text_output = self.proj(self.dropout(text_output)) + text_ori
        return text_output

class CoHead(nn.Module):
    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, embedding_dim_1=768, embedding_dim_2=512, dim=256, ft_out=256,fourier=True):
        super(CoHead, self).__init__()

        self.embedding_dim_1 = embedding_dim_1
        self.embedding_dim_2 = embedding_dim_2 
        self.k = dim 
        self.Wl = nn.Parameter(torch.Tensor((self.embedding_dim_1, self.k)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.embedding_dim_2)))
        self.Wi = nn.Parameter(torch.Tensor((self.k, self.embedding_dim_1)))
        self.wHi = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.dropout = nn.Dropout(0.3)
        self.final_proj = nn.Linear((embedding_dim_1 + embedding_dim_2)*2,  ft_out)
        self.disc = nn.Linear((embedding_dim_1 + embedding_dim_2)*2, 1)
        self.gelu = nn.GELU()


        #register weights and biAi Ai params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Wi", self.Wi)
        self.register_parameter("wHi", self.wHi)
        self.register_parameter("whc", self.whc)


        #initialize data of parameters
        self.Wl.data = torch.randn((self.embedding_dim_2, self.embedding_dim_1))
        self.Wc.data = torch.randn((self.k, self.embedding_dim_2))
        self.Wi.data = torch.randn((self.k, self.embedding_dim_1))
        self.wHi.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.fourier = fourier

    def forward(self, rep_1, rep_2, rep_1_ori, rep_2_ori, itm_head=False):
        if self.fourier:
            img_rep_fourier = torch.fft.fft2(rep_1).float()
            cap_rep_fourier = torch.fft.fft2(rep_2).float()
            img_rep_trans = img_rep_fourier.transpose(-1, -2)#[bs, dim, len]
            cap_rep_trans = cap_rep_fourier.transpose(-1, -2)#[bs, dim, len]
            L = self.gelu(torch.matmul(torch.matmul(img_rep_fourier, self.Wl), img_rep_trans))  
            L_trans = L.transpose(-1, -2)
        else:
            # print(rep_1.shape)
            # print(rep_2.shape)
            img_rep_trans = rep_1.transpose(-1, -2) #[bs, dim, len]
            cap_rep_trans = rep_2.transpose(-1, -2) #[bs, dim, len]
            L = torch.tanh(torch.matmul(torch.matmul(rep_2, self.Wl), img_rep_trans))  
            L_trans = L.transpose(-1, -2)

        Hi = torch.tanh(torch.matmul(self.Wi, img_rep_trans) + torch.matmul(torch.matmul(self.Wc, cap_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, cap_rep_trans) + torch.matmul(torch.matmul(self.Wi, img_rep_trans), L_trans))
        Ai = F.softmax(torch.matmul(self.wHi, Hi), dim=-1)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=-1)

        co_s = torch.matmul(Ai, rep_1).squeeze_(1) # (1, dim)
        co_c = torch.matmul(Ac, rep_2).squeeze_(1) # (1, dim)
        co_sc = torch.squeeze(torch.cat([rep_1_ori , co_s, co_c, rep_2_ori], dim = -1))
        if not itm_head:
            return self.final_proj(self.dropout(co_sc))
        else :
            return self.disc(self.dropout(co_sc))

class MixtureMultiModalLayer(nn.Module):
    def __init__(self, config:PerceiverConfig, d_visions=[256], d_texts=[256], num_self_attend=None) -> None:
        super().__init__()
        vision_layers = []
        text_layers = []
        self_attention_layers = []
        for dim in d_visions:
            vision_layers.append(PerceiverLayer(
                config,
                kv_dim=dim,
                q_dim=config.d_latents,
                is_cross_attention=True,
                use_query_residual=True,
                num_heads=config.num_cross_attention_heads,
                widening_factor=config.cross_attention_widening_factor,
            ))
        for dim in d_texts:
            text_layers.append(PerceiverLayer(
                config,
                kv_dim=dim,
                q_dim=config.d_latents,
                is_cross_attention=True,
                use_query_residual=True,
                num_heads=config.num_cross_attention_heads,
                widening_factor=config.cross_attention_widening_factor,
            ))
        self.vision_layers = nn.ModuleList(vision_layers)    
        self.text_layers = nn.ModuleList(text_layers)    

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
       
        text_outputs = []
        vision_outputs = []
        for i in range(len(text_inputs)): 
            text_outputs.append(self.text_layers[i](
                text_question,
                inputs=text_inputs[i],
            )[0])


        for i in range(len(vision_inputs)):
            vision_outputs.append(self.vision_layers[i](
                vision_question,
                inputs=vision_inputs[i],
            )[0])
        text_state = torch.cat(text_outputs, dim=1) 
        vision_state = torch.cat(vision_outputs, dim=1) 
        text_itm_state = text_state 
        vision_itm_state = vision_state 

        for layer_module in self.self_attends:
            input = torch.cat([text_state, vision_state], dim=1)
            input_itm = torch.cat([text_itm_state, vision_itm_state], dim=1)
            output = layer_module(input , attention_mask=self_attend_mask)[0]
            itm_state = layer_module(input_itm, attention_mask=None)[0]
            itm_state = itm_state + output
            text_state, vision_state = torch.split(output, [text_state.shape[1], vision_state.shape[1]] ,dim=1)
            text_itm_state, vision_itm_state = torch.split(itm_state, [text_state.shape[1], vision_state.shape[1]] ,dim=1)

        return text_state, vision_state, text_itm_state, vision_itm_state

    def get_vision_features(self, vision_inputs, question):
        vision_outputs = []
        for i in range(len(vision_inputs)):
            vision_outputs.append(self.vision_layers[i](
                question,
                inputs=vision_inputs[i],
            )[0])
        vision_state = torch.cat(vision_outputs, dim=1) 
        for _, layer_module in enumerate(self.self_attends):
            vision_state = layer_module(vision_state)[0]

        return vision_state

    def get_text_features(self, text_inputs, question ):
        text_outputs = []
        for i in range(len(text_inputs)): 
            text_outputs.append(self.text_layers[i](
                question,
                inputs=text_inputs[i],
            )[0])
        text_state = torch.cat(text_outputs, dim=1) 

        for _, layer_module in enumerate(self.self_attends):
            text_state= layer_module(text_state)[0]

        return text_state


class MixtureMultiModalHead(nn.Module):


    def __init__(self, config:PerceiverConfig, d_out, num_blocks, d_visions=[256], d_texts=[256]) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.text_question = nn.Parameter(torch.empty(config.num_latents, config.d_latents))
        self.vision_question = nn.Parameter(torch.empty(config.num_latents, config.d_latents))
        nn.init.kaiming_normal_(self.text_question, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.vision_question, mode='fan_out', nonlinearity='leaky_relu')

        multimodal_layers = [MixtureMultiModalLayer(config=config, d_texts=d_texts, d_visions=d_visions) for _ in range(self.num_blocks)]
        self.layers = nn.ModuleList(multimodal_layers)
        self.dropout= nn.Dropout(0.4) 
        self.proj = nn.Linear(config.d_latents, d_out)


    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, text_inputs, vision_inputs, self_attend_mask=None):
        bs = text_inputs[0].size(0)
        itm_vision= []
        itm_text = []
        text_question = self.text_question.expand([bs, -1, -1])
        vision_question = self.vision_question.expand([bs, -1, -1]) 

        for i in range(self.num_blocks):
            res_text = text_question
            res_vision = vision_question
            text_question, vision_question, itm_text_state, itm_vision_state = self.layers[i](
                text_inputs=text_inputs, 
                vision_inputs=vision_inputs, 
                text_question = text_question,
                vision_question = vision_question,
                self_attend_mask=self_attend_mask
            )
            text_question = text_question + res_vision
            vision_question = vision_question + res_text 

        itm_text = torch.mean(itm_text_state, dim=1)
        itm_vision = torch.mean(itm_vision_state, dim=1)

        text_ouput = torch.mean(text_question, dim=1) 
        vision_output = torch.mean(vision_question, dim=1)
        
        text_output = self.proj(self.dropout(text_ouput))
        vision_output = self.proj(self.dropout(vision_output))
        itm_text = self.proj(self.dropout(itm_text))
        itm_vision = self.proj(self.dropout(itm_vision))

        return text_output, vision_output, itm_text, itm_vision

    def get_vision_features(self, vision_inputs):
        bs = vision_inputs[0].size(0)
        text_question = self.text_question.expand([bs, -1, -1])
        vision_question = self.vision_question.expand([bs, -1, -1]) 

        for i in range(self.num_blocks):
            res_text = text_question
            vision_question = self.layers[i].get_vision_features(vision_inputs, vision_question)
            vision_question = vision_question + res_text 

        vision_output = torch.mean(vision_question, dim = 1)
        vision_output = self.proj(self.dropout(vision_output))
        return vision_output

    def get_text_features(self, text_inputs):
        bs = text_inputs[0].size(0)
        text_question = self.text_question.expand([bs, -1, -1])
        vision_question = self.vision_question.expand([bs, -1, -1]) 

        for i in range(self.num_blocks):
            res_vision = vision_question
            text_question = self.layers[i].get_text_features(text_inputs, text_question)
            text_question = text_question + res_vision 

        text_output = torch.mean(text_question, dim = 1)

        text_output = self.proj(self.dropout(text_output))
        return text_output