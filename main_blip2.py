from model.dctModel import CompressedLAVISBLIP2WithQueue 
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from trainer import MyTrainer as Blip2Trainer 
from utils.data_utils import  get_loaders
from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    from config import parser
    from config import POINCARE, EUCLID, LORENTZ, LAVIS_BLIP_BASE_FLICKR, LAVIS_BLIP_BASE_COCO, COCO, FLICKR
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
    FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"
    config = parser.parse_args()
    for dataset in [FLICKR]:
        config.dataset =dataset

        model, vis_processors, txt_processors = load_model_and_preprocess("blip2", "coco", is_eval=False)
        # tokenizer = model.tokenizer
        if "flickr" in config.dataset:
            config.model_ckt = LAVIS_BLIP_BASE_FLICKR
            dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
        else:
            config.model_ckt = LAVIS_BLIP_BASE_COCO 
            dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


        def inner_training_loop(batch_size):
            config.batch_size = batch_size
            train_loader, val_loader, test_loader, test_img2txt, test_txt2img, _, _ = get_loaders(
                20, 
                dataset,
                vis_processor=vis_processors['eval'],
                txt_processor=txt_processors['eval'],
                tokenizer=model.tokenizer,
                eval_batch_size=50
            )
            blip2_model = CompressedLAVISBLIP2WithQueue(config, model)

            trainer = Blip2Trainer(
                model=blip2_model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                img2txt=test_img2txt,
                txt2img=test_txt2img
            )
            print(trainer.evaluate())
            # print(trainer.evaluate('val'))
            # trainer.train()


        config.epochs = 2 
        config.enable_log = False
        config.use_margin_loss = False 

        for distil in [False]:
            config.distil = distil 
            # for compress_method in ['std', 'none']:
            # for compress_method in ['std','none', 'dct','mean','random']:
            for compress_method in [
                # 'std-mean-merge', 
                'PiToMe', 
                'ToMe',
                # 'std-mean-merge',
                'dct', 
            ]:
                config.compress_method = compress_method
                inner_training_loop(config.batch_size)
