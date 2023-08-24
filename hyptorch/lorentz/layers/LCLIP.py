from transformers.pytorch_utils import (
    prune_linear_layer,
    find_pruneable_heads_and_indices,
)
from transformers.activations import ACT2FN
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .LAttn import SelfAttention
from .LEmbed import LorentzEmbedding
from hyptorch.lorentz.layers.LFC import LorentzLinear
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block
from transformers import CLIPVisionConfig, CLIPTextConfig,CLIPModel
from hyptorch.lorentz.layers import (
    LorentzConv2d,
    LorentzBatchNorm2d,
)
from geoopt import ManifoldParameter
from hyptorch.lorentz.manifold import CustomLorentz
import torch.nn.functional as F


class CLIPSelfOutput(nn.Module):
    def __init__(self, manifold, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = LFC_Block(
            manifold,
            config.hidden_size,
            config.hidden_size,
            LFC_normalize=True,
            activation=ACT2FN[config.hidden_act],
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPAttention(nn.Module):
    def __init__(self, manifold ,config, position_embedding_type=None):
        super().__init__()
        self.self = SelfAttention(
            manifold=manifold,
            config=config,
            position_embedding_type=position_embedding_type,
        )
        self.output = CLIPSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class CLIPIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CLIPOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class VisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        print(patch_embeds.shape)
        patch_embeds = patch_embeds.flatten(2)
        print(patch_embeds.shape)
        patch_embeds = patch_embeds.transpose(1,2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings



class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, manifold:CustomLorentz,config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding =  ManifoldParameter(
            self.manifold.random_normal((self.embed_dim + 1)),
            manifold=self.manifold,
            requires_grad=True
        )

        self.patch_embedding = LorentzConv2d(
            manifold=manifold,
            in_channels=config.num_channels + 1,
            out_channels=self.embed_dim + 1,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
       
        self.batch_norm = LorentzBatchNorm2d(manifold=manifold, num_channels=self.embed_dim + 1)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = LorentzEmbedding(manifold, self.num_positions, self.embed_dim + 1)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        pixel_values = pixel_values.permute(0,2,3,1)
        pixel_values = F.pad(pixel_values, pad=(1,0), mode="constant", value=0)
        pixel_values = self.manifold.projx(pixel_values)

        self.manifold.assert_check_point_on_manifold(pixel_values)

        patch_embeds = self.batch_norm(self.patch_embedding(pixel_values))
        print(patch_embeds)
        # self.manifold.assert_check_point_on_manifold(patch_embeds)
        
        patch_embeds = self.manifold.lorentz_flatten(patch_embeds)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        # self.manifold.assert_check_point_on_manifold(class_embeds)

        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = embeddings.narrow(-1, 1, position_embeddings.shape[-1] - 1) + position_embeddings.narrow(
            -1, 1, position_embeddings.shape[-1] - 1
        )

        embeddings = self.manifold.add_time(embeddings)


        return embeddings


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, manifold:CustomLorentz ,config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.manifold = manifold
        self.token_embedding = LorentzEmbedding(manifold, config.vocab_size, embed_dim+1)
        self.position_embedding = LorentzEmbedding(manifold, config.max_position_embeddings, embed_dim+1)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        inputs_embeds = inputs_embeds.narrow(-1, 1, position_embeddings.shape[-1] - 1) + position_embeddings.narrow(
            -1, 1, position_embeddings.shape[-1] - 1
        )
        inputs_embeds = self.manifold.add_time(inputs_embeds)


        return inputs_embeds
