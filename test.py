import torch
from transformers import (
    CLIPProcessor,
)
from datasets import load_dataset
from model.hypCLIP import HypCLIP
from model.hypBLIP import HypBLIP
# from model.hybridCLIP import HypCLIP as HybridCLIP
# from model.PoincareCLIP import  PoincareCLIP 
from model.perceiverMixtureModel import MyModel
# from model.perceiverMixtureModel import MyModel
from transformers import CLIPProcessor, BlipProcessor
from utils.data_utils import get_co_dataloader, co_preprocess_img 
from coTrainer import MyTrainer
from accelerate import find_executable_batch_size
from utils.data_utils import get_flickr


if __name__ == "__main__":
    # from config import parser
    # from config import EUCLID, LORENTZ, POINCARE 
    # from config import EUCLID, LORENTZ, POINCARE 
    # from config import CLIP_BASE_PATCH_16, BLIP_BASE_FLICKR 

    # config = parser.parse_args()
    # print("Getting BLIP processor...")
    # clip_processor = CLIPProcessor.from_pretrained(
    #     CLIP_BASE_PATCH_16, cache_dir=config.cache_dir
    # )
    # print("Getting CLIP processor...")
    # blip_processor = BlipProcessor.from_pretrained(
    #     BLIP_BASE_FLICKR, cache_dir=config.cache_dir
    # )

    # if "flickr" in config.dataset:
    #     dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)
    # else:
    #     dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)

    # dataset = dataset.map(
    #     lambda sample: co_preprocess_img(sample, clip_processor=clip_processor, blip_processor=blip_processor)
    # ).remove_columns(["image"])
    # dataset.set_format("numpy")


    # @find_executable_batch_size(starting_batch_size=config.batch_size)
    # def inner_training_loop(batch_size):
    #     config.batch_size = batch_size
    #     train_loader = get_co_dataloader(
    #         dataset["train"],
    #         config.batch_size,
    #         clip_processor=clip_processor,
    #         blip_processor=blip_processor,
    #         mode="train",
    #         use_random_sampler=False,
    #     )
    #     test_loader = get_co_dataloader(
    #         dataset["test"], 5, 
    #         blip_processor=blip_processor, 
    #         clip_processor=clip_processor,
    #         mode="test"
    #     )
    #     val_loader = get_co_dataloader(
    #         dataset["val"], 
    #         5, 
    #         blip_processor=blip_processor, 
    #         clip_processor=clip_processor,
    #         mode="val"
    #     )
    #     model = MyModel(config) 
    #     trainer = MyTrainer(
    #         model=model,
    #         config=config,
    #         dataset=dataset,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         test_loader=test_loader,
    #         clip_processor=clip_processor,
    #         blip_processor=blip_processor,
    #     )
    #     trainer.train()
    # for curv in [5.0, 10.0]:
    #     config.curv = curv
    #     for manifold in [EUCLID]:
    #         config.manifold = manifold
    #         inner_training_loop()
    import tensorly as tl
    tl.set_backend('pytorch')
    matrix = torch.rand(100, 224, 768)
    U, S, V = torch.linalg.svd(matrix, full_matrices=False)
    R = U @ torch.diag_embed(S) @ V
    print(torch.dist(matrix, R) + R)


    
    
