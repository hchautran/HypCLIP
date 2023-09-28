import torch
import torch.nn as nn
from .modules.text_model import CLIPText
from .modules.vision_model import CLIPVision
from loralib.lora_clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from loralib.utils import mark_only_lora_as_trainable 

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
        mark_only_lora_as_trainable(model=text_model)
        mark_only_lora_as_trainable(model=vision_model)
        


        text_body = text_model.text_model
        vision_body = vision_model.vision_model
        text_head = nn.ModuleList([])
        vision_head = nn.ModuleList([])
        text_head = nn.ModuleList([text_model.text_projection])
        vision_head = nn.ModuleList([vision_model.visual_projection])

        if self.manifold_name !=  EUCLID:
            if config.use_lorentz_centroid:
                text_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
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
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
                )
                vision_head.append(
                    ManifoldMapper(self.manifold, curv=self.curv, clip_r=self.clip_r, use_normalize=False)
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
