from model.hypBLIP2 import Blip2Model 
from model.perceiverModel import PerceiverCLIPWithQueue 
from model.hypCLIP import HypCLIP 
from lavis.datasets.builders import load_dataset
from trainer import MyTrainer as Trainer 
from trainer_perceiver import MyTrainer as PerceiverTrainer
from utils.data_utils import  get_loaders
from lavis.models import load_model_and_preprocess

from transformers import (
    CLIPProcessor,
)

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
    FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"

    config = parser.parse_args()
    # model, vis_processors, txt_processors = load_model_and_preprocess("blip2", "coco", is_eval=False)
    # tokenizer = model.tokenizer
    processor = CLIPProcessor.from_pretrained(
        config.model_ckt, cache_dir=config.cache_dir
    )
    if "flickr" in config.dataset:
        # config.model_ckt = LAVIS_BLIP_BASE_FLICKR
        dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    else:
        # config.model_ckt = LAVIS_BLIP_BASE_COCO 
        dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


    def inner_training_loop(batch_size):
        config.batch_size = batch_size
        train_loader, val_loader, test_loader= get_loaders(
            config, 
            dataset,
            vis_processor=processor,
            txt_processor=None,
            tokenizer=processor,
        )


        # blip2_model = Blip2Model(config, model) 
        # trainer = TrainerWithQueue(
        #     model=blip2_model,
        #     config=config,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     test_loader=test_loader,
        # )

        blip2_model = PerceiverCLIPWithQueue(config) 
        trainer = PerceiverTrainer(
            model=blip2_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        trainer.train()
        itc_metrics= trainer.evaluate('test')
        print(itc_metrics)
        # print(itm_metrics)
        itc_metrics= trainer.evaluate('val')
        print(itc_metrics)
        # trainer.train()
        # print(itm_metrics)



    config.epochs = 100 
    config.enable_log = False 
    inner_training_loop(config.batch_size)
    # for curv in [1.0, 2.0, 10.0]:
    #     config.curv = curv
    #     for margin_loss in [True, False]:
    #         config.margin_loss = margin_loss 
    #         for use_graph in [True, False]:
    #             config.use_graph=use_graph
    #             inner_training_loop(config.batch_size)