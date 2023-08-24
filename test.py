from torch import embedding_renorm_
from LAVIS.evaluate import main
from hyptorch.lorentz.layers.LEmbed import LorentzEmbedding
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LCLIP import CLIPTextEmbeddings, CLIPVisionEmbeddings, VisionEmbeddings 
from hyptorch.lorentz.layers.LFC import LorentzLinear 
from transformers import CLIPConfig 
import torch
from transformers import (
    CLIPProcessor,
)
from transformers import CLIPProcessor, BlipProcessor
from utils.data_utils import get_dataloader, preprocess_img
from utils.data_utils import get_flickr


if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE

    config = parser.parse_args()
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
    manifold = CustomLorentz(k=0.1)
    text_emb = CLIPTextEmbeddings(manifold=manifold, config=clip_cfg.text_config)
    vision_emb = CLIPVisionEmbeddings(manifold=manifold, config=clip_cfg.vision_config)
    # vision_emb = VisionEmbeddings(config=clip_cfg.vision_config)

    x = torch.rand((5, 100))
    x = manifold.projx(manifold.expmap0(x))
    manifold.assert_check_point_on_manifold(x)
    linear = LorentzLinear(manifold, 100, 100) 
    x = linear(x)
    manifold.assert_check_point_on_manifold(x)

    for batch in train_loader: 
        print(batch['pixel_values'].shape)
        print(batch['input_ids'].shape)
        text =text_emb(batch['input_ids'])
        print('text size', text.shape)
        manifold.assert_check_point_on_manifold(text)
        img =vision_emb(batch['pixel_values'])
        print(img)
        manifold.assert_check_point_on_manifold(img)
        
        break




