import torch
import torch.nn as nn
from .modules.text_model import CLIPText
from .modules.vision_model import CLIPVision
from .modules.discriminator import Discriminator as DisModel
from .modules.seq_linear import LorentzSeqLinear, HypSeqLinear
from .modules.hyp_discriminator import HypDiscriminator as HypDisModel
from .modules.hyp_discriminator import LorentzDiscriminator as LorentzDisModel
from .manifolds.euclidean import Euclidean
from .manifolds.hyperboloid import Hyperboloid
from .manifolds.lorentz import Lorentz
from .manifolds.poincare import PoincareBall
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from transformers.models.clip.modeling_clip import CLIPOutput
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block
from .modules.utils import ManifoldMapper, LorentzCentroidPooler
from model.baseModel import BaseModel
from peft import get_peft_model, LoraConfig, TaskType

EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypCLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypCLIP, self).__init__(config)

        text_model = CLIPTextModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        print(vision_model)
        text_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['k_proj', 'v_proj', 'q_proj', 'fc1', 'fc2', 'text_projection']
        )
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=4, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=[
                'visual_projection', 
                '*.11.mlp.fc1', 
                '*.11.mlp.fc2', 
                '*.11.self_attn.out_proj',
                '*.11.self_attn.q_proj',
                '*.11.self_attn.k_proj',
                '*.11.self_attn.v_proj',
                ] 
        )
        text_model = get_peft_model(text_model, text_peft_config)
        print(text_model.print_trainable_parameters())
        vision_model = get_peft_model(vision_model, vision_peft_config)
        print(vision_model.print_trainable_parameters())

        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = nn.ModuleList([])
        vision_head = nn.ModuleList([])
        text_head = nn.ModuleList([text_model.text_projection])
        vision_head = nn.ModuleList([vision_model.visual_projection])

        if self.manifold_name == LORENTZ:
            if config.use_lorentz_centroid:
                text_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
                )
                text_head.append(
                    LorentzCentroidPooler(
                        self.manifold, curv=self.curv, clip_r=self.clip_r
                    )
                )
                vision_head.append(
                    LorentzCentroidPooler(
                        self.manifold, curv=self.curv, clip_r=self.clip_r
                    )
                )
            else:
                text_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r)
                )


        self.vision_model = CLIPVision(
            config=config,
            body=vision_body,
            head=vision_head,
            num_trainable_blocks=config.vision_trainable_blocks,
            freeze_embedding=config.freeze_embedding,
        )
        self.text_model = CLIPText(
            config=config,
            body=text_body,
            head=text_head,
            num_trainable_blocks=config.text_trainable_blocks,
            freeze_embeddings=config.freeze_embedding,
        )
