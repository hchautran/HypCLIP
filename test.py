from model.modules.fuseModel import FuseEncoder 
from lavis.datasets.builders import load_dataset
from model.modules.fuseModel import LavisEncoder, CLIPEncoder, BLIPEncoder 
from utils.data_utils import get_fused_dataloader 
from lavis.models import load_model_and_preprocess
from hyptorch.lorentz.manifold import CustomLorentz
from transformers import AutoProcessor, AutoModel 
from lavis import BlipRetrieval
from transformers import CLIPModel, BlipModel 
from model.modules.utils import prepare_encoder, prepare_processors_and_models
from tqdm.auto import tqdm
from config import parser
from typing import List
from config import (
    CLIP_LARGE_PATCH_14, 
    CLIP_BASE_PATCH_32, 
    CLIP_BASE_PATCH_16, 
    LAVIS_BLIP_BASE_FLICKR, 
    BLIP_BASE_FLICKR,
    BLIP_BASE_COCO,
    LAVIS_BLIP_BASE_COCO, 
    COCO_PATH, 
    FLICKR_PATH
)

config = parser.parse_args()

if __name__ == '__main__':
    dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    model_ckts = [LAVIS_BLIP_BASE_COCO, CLIP_BASE_PATCH_32, BLIP_BASE_COCO]


    tokenizers, vis_processors, txt_processors, models = prepare_processors_and_models(model_ckts)
    # vis_encoders, text_encoders, text_head, vision_head = prepare_encoder(config, models)
    print('--'*50)
    print(models[0])
    print('--'*50)
    print(models[1])
    print('--'*50)
    print(models[2])
            

    train_loaders = get_fused_dataloader(dataset, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, batch_size=40, mode='train') 
    test_loader= get_fused_dataloader(dataset, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, batch_size=1, mode='test') 
    for batch in tqdm(test_loader):
        print(batch[f'pixel_values_{0}'].shape)
        print(batch[f'pixel_values_{1}'].shape)
        print(batch[f'input_ids_{0}'].shape)
        print(batch[f'input_ids_{1}'].shape)
        print(batch[f'attention_mask_{0}'].shape)
        print(batch[f'attention_mask_{1}'].shape)
        break

    # model = FuseEncoder(
    #     config,           
    #     d_visions=[100, 100], 
    #     d_texts=[100, 100], 
    #     ft_out=256, 
    #     vision_bodies=[], 
    #     text_bodies=[],
    #     vision_head=None,
    #     text_head=None,
    #     mapper=None,
    #     manifold=None, 
    # )