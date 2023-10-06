import torch
import torch.nn as nn
from .modules.text_model import BLIPText
from .modules.vision_model import BLIPVision
from transformers import BlipForImageTextRetrieval
from .modules.utils import ManifoldMapper
from model.baseModel import BaseModel
from .modules.utils import ManifoldMapper, LorentzCentroidPooler
from transformers.activations import ACT2FN
from peft import get_peft_model, LoraConfig, TaskType
from torch_geometric.nn import GATv2Conv


EUCLID = "euclidean"
POINCARE = "poincare"
LORENTZ = "lorentz"


class HypBLIP(BaseModel):
    def __init__(self, config) -> None:
        super(HypBLIP, self).__init__(config)

        model = BlipForImageTextRetrieval.from_pretrained(
            self.model_ckt, cache_dir=config.cache_dir
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1, 
            target_modules=[
                'dense', 
                'query', 
                'value',
                'key', 
                'text_proj', 
                'vision_proj',
                '*.11.self_attn.qkv'
                '*.11.self_attn.projection'
                '*.11.mlp.fc1'
                '*.11.mlp.fc2'
            ]
        )
        print(model)
        model = get_peft_model(model, peft_config) 
        text_body = model.text_encoder
        vision_body = model.vision_model
        text_head = nn.ModuleList([model.text_proj])
        vision_head = nn.ModuleList([model.vision_proj])
        if self.manifold_name == LORENTZ:
            if config.use_lorentz_centroid:
                text_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=True)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=True)
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
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=True)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=True)
                )


        self.vision_model = BLIPVision(
            config,
            body=vision_body,
            head=vision_head,
            # num_trainable_blocks=config.vision_trainable_blocks,
            # freeze_embedding=config.freeze_embedding,
        )
        self.text_model = BLIPText(
            config,
            body=text_body,
            head=text_head,
            # num_trainable_blocks=config.text_trainable_blocks,
            # freeze_embeddings=config.freeze_embedding,
        )
