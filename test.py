from torch import embedding_renorm_
from LAVIS.evaluate import main
from hyptorch.lorentz.layers.LEmbed import LorentzEmbedding
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LCLIP import CLIPTextEmbeddings, CLIPVisionEmbeddings, VisionEmbeddings 
from hyptorch.lorentz.layers.LFC import LorentzLinear 
from hyptorch.lorentz.blocks.layer_blocks import LFC_Block 
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
    x = torch.rand((100, 384)).uniform_(-1.5, 1.5) 
    x = manifold.expmap0(F.pad(x, (1,0), 'constant', 0))
    print(manifold.inner(x, x))
   
    manifold.assert_check_point_on_manifold(x)
    layers = nn.Sequential(
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
        LFC_Block(manifold, in_features=x.shape[-1], out_features=x.shape[-1], activation=ACT2FN["gelu"], normalization="batch_norm"),
    )

    x = layers(x)
    manifold.assert_check_point_on_manifold(x)
    # print(manifold.inner(x, x))







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


    train_loader = get_dataloader(
        dataset["train"],
        config.batch_size,
        processor=processor,
        mode="train",
        use_random_sampler=False,
    )
    test_loader = get_dataloader(dataset["test"], 5, processor=processor, mode="test")
    val_loader = get_dataloader(dataset["val"], 5, processor=processor, mode="val")

    
    clip_cfg = CLIPConfig().from_pretrained(config.model_ckt)
    manifold = CustomLorentz(k=config.curv)
    text_emb = CLIPTextEmbeddings(manifold=manifold, config=clip_cfg.text_config)
    vision_emb = CLIPVisionEmbeddings(manifold=manifold, config=clip_cfg.vision_config)
    # vision_emb = VisionEmbeddings(config=clip_cfg.vision_config)




    for batch in train_loader: 
        print(batch['pixel_values'].shape)
        print(batch['input_ids'].shape)
        text =text_emb(batch['input_ids'])
        print('text size', text.shape)
        manifold.assert_check_point_on_manifold(text)
        img =vision_emb(batch['pixel_values'])
        print(img.shape)
        print(manifold._check_point_on_manifold(img))
        manifold.assert_check_point_on_manifold(img)
        
        break




