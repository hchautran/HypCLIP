from transformers.pytorch_utils import (
    prune_linear_layer,
    find_pruneable_heads_and_indices,
)
from transformers.activations import ACT2FN
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .LEmbed import LorentzEmbedding
from hyptorch.lorentz.layers.LFC import LorentzLinear
from transformers import CLIPVisionConfig, CLIPTextConfig
from hyptorch.lorentz.layers import (
    LorentzBatchNorm1d,
    LorentzBatchNorm2d,
    LorentzConv2d,
)
from geoopt import ManifoldParameter
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LFC import LorentzLinear 
from transformers import CLIPConfig
import torch.nn.functional as F

class HypCLIPVisionEmbeddings(nn.Module):
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
        self.num_positions = self.num_patches
        self.position_embedding = LorentzEmbedding(manifold, self.num_positions, self.embed_dim + 1)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        pixel_values = pixel_values.permute(0,2,3,1)
        pixel_values = F.pad(pixel_values, pad=(1,0), mode="constant", value=0)
        pixel_values = self.manifold.expmap0(pixel_values)


        patch_embeds = self.batch_norm(self.patch_embedding(pixel_values))
        embeddings = self.manifold.lorentz_flatten(patch_embeds)

        
        self.manifold.assert_check_point_on_manifold(pixel_values)
        self.manifold.assert_check_point_on_manifold(patch_embeds)
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = embeddings.narrow(-1, 1, position_embeddings.shape[-1] - 1) + position_embeddings.narrow(
            -1, 1, position_embeddings.shape[-1] - 1
        )

        embeddings = self.manifold.add_time(embeddings)


        return embeddings


class HypCLIPTextEmbeddings(nn.Module):
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


class HypCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, manifold:CustomLorentz, config):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.bias = nn.Parameter(torch.zeros(()))

        self.k_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1)
        self.v_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1)
        self.q_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1)
        self.out_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        space = tensor.narrow(-1, 1, tensor.shape[-1] - 1)
        space = space.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        tensor = self.manifold.add_time(space) 
        return tensor 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) 
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim + 1)

        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        self.manifold.assert_check_point_on_manifold(query_states)
        self.manifold.assert_check_point_on_manifold(key_states)
        self.manifold.assert_check_point_on_manifold(value_states)

        src_len = key_states.size(1)
        attn_weights = (
            2 + 2 * self.manifold.bmm(query_states, key_states.transpose(1, 2))
        ) * self.scale + self.bias

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = self.manifold.centroid(value_states, attn_probs)
        self.manifold.assert_check_point_on_manifold(attn_output)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim + 1):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim + 1)
        attn_output = attn_output.transpose(1, 2)
        space = attn_output.narrow(-1, 1, attn_output.shape[-1] - 1)
        attn_output = space.reshape(bsz, tgt_len, embed_dim - 1)

        attn_output = self.manifold.add_time(attn_output)
        self.manifold.assert_check_point_on_manifold(attn_output)

        attn_output = self.out_proj(attn_output)
        self.manifold.assert_check_point_on_manifold(attn_output)

        return attn_output, attn_weights_reshaped


class HypCLIPMLP(nn.Module):
    def __init__(self, manifold: CustomLorentz, config):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = LorentzLinear(manifold, config.hidden_size + 1, config.intermediate_size + 1)
        self.fc2 = LorentzLinear(manifold, config.intermediate_size + 1, config.hidden_size + 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.manifold.lorentz_activation(hidden_states, self.activation_fn)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class HypCLIPEncoderLayer(nn.Module):
    def __init__(self, manifold:CustomLorentz, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = HypCLIPAttention(manifold, config)
        self.batch_norm1 = LorentzBatchNorm1d(manifold, self.embed_dim+1)
        self.batch_norm2 = LorentzBatchNorm1d(manifold, self.embed_dim+1)
        self.mlp = HypCLIPMLP(manifold, config)
        self.manifold = manifold

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.batch_norm1(hidden_states)
        self.manifold.assert_check_point_on_manifold(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        self.manifold.assert_check_point_on_manifold(hidden_states)
        
        # residual connection
        hidden_states = hidden_states.narrow(-1, 1, residual.shape[-1] - 1) + residual.narrow(
            -1, 1, residual.shape[-1] - 1
        )
        hidden_states = self.manifold.add_time(hidden_states)


        residual = hidden_states

        hidden_states = self.batch_norm2(hidden_states)
        self.manifold.assert_check_point_on_manifold(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.manifold.assert_check_point_on_manifold(hidden_states)

        # residual connection
        hidden_states = hidden_states.narrow(-1, 1, residual.shape[-1] - 1) + residual.narrow(
            -1, 1, residual.shape[-1] - 1
        )
        hidden_states = self.manifold.add_time(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


