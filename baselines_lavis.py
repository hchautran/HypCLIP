from model.hypBLIP import LavisBLIPWithQueue ,LavisHypGraphBLIPWithQueue
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from utils.data_utils import  get_loaders
from tqdm.auto import tqdm

from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
    FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"

    config = parser.parse_args()
    model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
    tokenizer = model.tokenizer
    if "flickr" in config.dataset:
        config.model_ckt = LAVIS_BLIP_BASE_FLICKR
        dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    else:
        config.model_ckt = LAVIS_BLIP_BASE_COCO 
        dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)



    def inner_training_loop(batch_size):

        config.batch_size = batch_size

        train_loader, val_loader, test_loader= get_loaders(
            config, 
            dataset,
            vis_processor=vis_processors['eval'],
            txt_processor=txt_processors['eval'],
            tokenizer=model.tokenizer,
        )

        queue_model = LavisBLIPWithQueue(config, model) if not config.use_graph else LavisHypGraphBLIPWithQueue(config, model)
        trainer = LavisTrainer(
            model=queue_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        # print(trainer.evaluate('test'))
        # print(trainer.evaluate('val'))
        trainer.train()

    config.epochs = 3 
    config.enable_log = True 
    config.manifold = EUCLID
    config.use_entailment_loss = False 
    for use_margin_loss in [False, True]:
        config.use_margin_loss = use_margin_loss
        for use_graph in [False, True]:
            config.use_graph=use_graph
            inner_training_loop(config.batch_size)
