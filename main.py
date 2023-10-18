import torch
from transformers import (
    CLIPProcessor,
)
from datasets import load_dataset
from model.hypCLIP import HypCLIP, HypGraphCLIP, HypGraphCLIPWithQueue, HypCLIPWithQueue 
from model.hypBLIP import HypBLIPWithQueue, HypGraphBLIPWithQueue
from model.perceiverModel import MyModel
from transformers import CLIPProcessor, BlipProcessor
from trainer_lavis import MyTrainer as LavisTrainer
from utils.data_utils import get_dataloader, preprocess_img
from trainer import MyTrainer
from accelerate import find_executable_batch_size
from utils.data_utils import get_flickr


if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE 
    from config import CLIP_BASE_PATCH_16, BLIP_BASE_FLICKR, CLIP_LARGE_PATCH_14
    config = parser.parse_args()
    for model_ckt in [CLIP_LARGE_PATCH_14]:
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
            dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)
        else:
            dataset = get_flickr(config.dataset, cache_dir=config.cache_dir)

        dataset = dataset.map(
            lambda sample: preprocess_img(sample, processor=processor)
        ).remove_columns(["image"])
        dataset.set_format("numpy")


        @find_executable_batch_size(starting_batch_size=config.batch_size)
        def inner_training_loop(batch_size):
            config.batch_size = batch_size
            train_loader = get_dataloader(
                dataset["train"],
                config.batch_size,
                processor=processor,
                mode="train",
                use_random_sampler=False,
            )
            test_loader = get_dataloader(
                dataset["test"], 5, processor=processor, mode="test"
            )
            val_loader = get_dataloader(dataset["val"], 5, processor=processor, mode="val")
            if config.use_graph:
                model = HypGraphCLIPWithQueue(config) if "clip" in config.model_ckt else HypGraphBLIPWithQueue(config)
            else:
                model = HypCLIPWithQueue(config) if "clip" in config.model_ckt else HypBLIPWithQueue(config)
            # model = HypCLIP(config) if "clip" in config.model_ckt else HypBLIP(config)
            trainer = LavisTrainer(
                model=model,
                config=config,
                dataset=dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                processor=processor,
            )
            # print(trainer.evaluate())
            trainer.train()

        config.epochs = 6 
        config.enable_log = True 
        config.model_ckt = model_ckt
        for use_graph in [False, True]:
            config.use_graph = use_graph
            for manifold in [LORENTZ, EUCLID]:
                config.manifold = manifold
                inner_training_loop()
