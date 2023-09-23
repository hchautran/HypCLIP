import torch
from transformers import (
    CLIPProcessor,
)
from datasets import load_dataset
# from model.hypCLIP import HypCLIP
# from model.hypBLIP import HypBLIP
from model.hypSwin import HypSwin 
from model.hybridBLIP import HypBLIP 
# from model.PoincareCLIP import  PoincareCLIP 
from model.perceiverModel import MyModel
# from model.perceiverMixtureModel import MyModel
from transformers import CLIPProcessor, BlipProcessor, AutoTokenizer, AutoImageProcessor
from utils.data_utils import get_dataloader, preprocess_img
from trainer import MyTrainer
from accelerate import find_executable_batch_size
from utils.data_utils import get_flickr


if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE 
    from config import EUCLID, LORENTZ, POINCARE, SWIN_V2_BASE 

    config = parser.parse_args()
 
    print("Getting swin processor...")
    processor = AutoImageProcessor.from_pretrained(
        SWIN_V2_BASE, cache_dir=config.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckt)

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
            processor=tokenizer,
            mode="train",
            use_random_sampler=False,
        )
        test_loader = get_dataloader(
            dataset["test"], 5, processor=processor, mode="test"
        )
        val_loader = get_dataloader(dataset["val"], 5, processor=processor, mode="val")
        # model = HypCLIP(config) if "clip" in config.model_ckt else HypBLIP(config)
        model = HypSwin(config) 
        trainer = MyTrainer(
            model=model,
            config=config,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            processor=processor,
        )
        trainer.train()

    for manifold in [EUCLID]:
        config.manifold = manifold
        inner_training_loop()
