import torch
from model.hypCLIP import HypGraphCLIPWithQueue, HypCLIPWithQueue 
from model.hypBLIP import HypBLIPWithQueue, HypGraphBLIPWithQueue

from lavis import CLIP
from model.hypBLIP import LavisBLIPWithQueue ,LavisHypGraphBLIPWithQueue
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from utils.data_utils import  get_loaders
from tqdm.auto import tqdm

from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, LAVIS_BLIP_BASE_COCO, LAVIS_BLIP_BASE_FLICKR
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
    FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"

    config = parser.parse_args()
    model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
    tokenizer = model.tokenizer
    # dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)
    dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    train_loader, val_loader, test_loader= get_loaders(
        config, 
        dataset,
        vis_processor=vis_processors['eval'],
        txt_processor=txt_processors['eval'],
        tokenizer=model.tokenizer,
    )
    # for batch in tqdm(train_loader):
    #     print(batch['pixel_values'].shape)
    # for batch in tqdm(test_loader):
    #     print(batch['pixel_values'].shape)


    def inner_training_loop(batch_size):
        config.batch_size = batch_size
        config.model_ckt = LAVIS_BLIP_BASE_FLICKR
        queue_model = LavisBLIPWithQueue(config, model) if not config.use_graph else LavisHypGraphBLIPWithQueue(config, model)
        trainer = LavisTrainer(
            model=queue_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        # metric = trainer.evaluate(mode='test')
        # print(metric)
        # metric = trainer.evaluate(mode='val')
        # print(metric)
        trainer.train()

    config.epochs = 5 
    config.enable_log = True 
    for curv in [1.0, 2.0, 10.0]:
        config.curv = curv
        for use_entailment_loss in [False]:
            config.use_entailment_loss = use_entailment_loss
            for margin_loss in [True, False]:
                config.margin_loss = margin_loss 
                for use_graph in [True, False]:
                    config.use_graph=use_graph
                    for manifold in [LORENTZ]:
                        config.manifold = manifold 
                        inner_training_loop(config.batch_size)