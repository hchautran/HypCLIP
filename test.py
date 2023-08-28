from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block 
from hyptorch.models.clip import (
    HypCLIPVisionTransformer,
    HypCLIPTextTransformer,
    HypCLIPModel
) 
from transformers import CLIPConfig 
import torch
from transformers import (
    CLIPProcessor,
)
from transformers import CLIPProcessor, BlipProcessor
from utils.data_utils import get_dataloader, preprocess_img
from utils.data_utils import get_flickr
import torch.nn.functional  as F
import torch.nn as nn
from transformers.activations import ACT2FN

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE
    config = parser.parse_args()
    manifold = CustomLorentz(k=config.curv, atol=config.atol, rtol=config.rtol)
    x = torch.rand((100, 512)).uniform_(-1.0, 1.0) 
    x = manifold.expmap0(F.pad(x, (1,0), 'constant', 0))
   
    manifold.assert_check_point_on_manifold(x)
    layers = nn.Sequential(
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
    )

    x = layers(x)
    manifold.assert_check_point_on_manifold(x)

    if "blip" in config.model_ckt:
        print("Getting BLIP processor...")
        processor = BlipProcessor.from_pretrained(
            config.model_ckt, cache_dir=config.cache_dir
        )
    else:
        print("Getting CLIP processor...")
        processor = CLIPProcessor.from_pretrained(
            config.model_ckt, cache_dir=config.cache_dir
        )

    if "flickr" in config.dataset:
        dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)
    else:
        dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)

    dataset = dataset.map(
        lambda sample: preprocess_img(sample, processor=processor)
    ).remove_columns(["image"])
    dataset.set_format("numpy")


    test_loader = get_dataloader(dataset["test"], 5, processor=processor, mode="test")

    clip_cfg = CLIPConfig.from_pretrained(config.model_ckt)
    manifold = CustomLorentz(k=config.curv)
    text_model = HypCLIPTextTransformer(manifold=manifold, config=clip_cfg.text_config)
    vision_model = HypCLIPVisionTransformer(manifold=manifold, config=clip_cfg.vision_config)
    model = HypCLIPModel(manifold=manifold, config=clip_cfg)

    for batch in test_loader: 
        text =text_model(batch['input_ids'], batch['attention_mask'])
        manifold.assert_check_point_on_manifold(text[0])
        vision =vision_model(batch['pixel_values'])
        manifold.assert_check_point_on_manifold(vision[0])
        out = model(
            
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask']
        )
        print(out)
        
        break




