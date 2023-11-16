import torch
from transformers import (
    CLIPProcessor,
)
from lavis.datasets.builders import load_dataset
from model.hypCLIP import HypGraphCLIPWithQueue, HypCLIPWithQueue 
from model.hypBLIP import HypBLIPWithQueue, HypGraphBLIPWithQueue
from transformers import CLIPProcessor, BlipProcessor
from trainer_queue import MyTrainer as LavisTrainer
from accelerate import find_executable_batch_size
from utils.data_utils import get_loaders 


if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ
    from config import CLIP_BASE_PATCH_16, CLIP_BASE_PATCH_32, COCO_PATH, FLICKR_PATH 
    config = parser.parse_args()
    for model_ckt in [CLIP_BASE_PATCH_16, CLIP_BASE_PATCH_32]:
        config.model_ckt = model_ckt
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
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)



        def inner_training_loop(batch_size):
            config.batch_size = batch_size
            train_loader, val_loader, test_loader = get_loaders(
                config, 
                dataset,
                vis_processor=processor,
                txt_processor=None,
                tokenizer=processor,
            )

            if config.use_graph:
                model = HypGraphCLIPWithQueue(config) if "clip" in config.model_ckt else HypGraphBLIPWithQueue(config)
            else:
                model = HypCLIPWithQueue(config) if "clip" in config.model_ckt else HypBLIPWithQueue(config)
            trainer = LavisTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            # print(trainer.evaluate('test'))
            # print(trainer.evaluate('val'))
            trainer.train()

        config.epochs = 5 
        config.enable_log = True 
        config.manifold = LORENTZ 
        config.use_entailment_loss = False 
        for curv in [1.0, 2.0, 10.0]:
            config.curv = curv
            for margin_loss in [True, False]:
                config.margin_loss = margin_loss 
                for use_graph in [True, False]:
                    config.use_graph=use_graph
                    inner_training_loop(config.batch_size)
