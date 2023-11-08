import torch
from model.hypCLIP import HypGraphCLIPWithQueue, HypCLIPWithQueue 
from model.hypBLIP import HypBLIPWithQueue, HypGraphBLIPWithQueue

from model.hypBLIP import LavisBLIPWithQueue ,LavisHypGraphBLIPWithQueue
from lavis.datasets.builders import load_dataset
# from model.hypCLIP import HypGraphCLIPWithQueue 
from utils.data_utils import get_coco_dataloader, get_coco 
from tqdm.auto import tqdm

from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE, LAVIS_BLIP_BASE_FLICKR
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"

    config = parser.parse_args()
    model, vis_processors, txt_processors = load_model_and_preprocess("blip_retrieval", "coco", is_eval=False)
    tokenizer = model.tokenizer
    coco_dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)
    train_loader  = get_coco_dataloader(
        coco_dataset=coco_dataset,
        batch_size=config.batch_size,
        vis_processor=vis_processors['eval'],
        txt_processor=txt_processors['eval'],
        tokenizer=model.tokenizer,
        mode='train',
    ) 
    test_loader  = get_coco_dataloader(
        coco_dataset=coco_dataset,
        batch_size=5,
        vis_processor=vis_processors['eval'],
        txt_processor=txt_processors['eval'],
        tokenizer=model.tokenizer,
        mode='test',
    ) 
    val_loader  = get_coco_dataloader(
        coco_dataset=coco_dataset,
        batch_size=5,
        vis_processor=vis_processors['eval'],
        txt_processor=txt_processors['eval'],
        tokenizer=model.tokenizer,
        mode='val',
    ) 
    print(coco_dataset['val'])
    print(coco_dataset['train'])
    print(coco_dataset['test'])
    print(len(coco_dataset['test'].img2txt))
    print(len(coco_dataset['test'].image))

    for i in range(len(coco_dataset['test'].text)):
        img_index = coco_dataset['test'].txt2img[i]
        img = coco_dataset['test'][img_index] 
        print(coco_dataset['test'].text[i]
        print(img)

        # break

    # for batch in tqdm(test_loader):
        # print(batch['pixel_values'].shape)
        # print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        # print(batch['img_id'].shape)
        # break
    

    # dataset = dataset.map(
        # lambda sample: lavis_preprocess_img(sample, processor=vis_processors['eval'])
    # ).remove_columns(["image"])
    # dataset.set_format("numpy")
    


    @find_executable_batch_size(starting_batch_size=config.batch_size)
    def inner_training_loop(batch_size):
        config.batch_size = batch_size

        config.model_ckt = LAVIS_BLIP_BASE_FLICKR 
        # queue_model = LavisBLIPWithQueue(config, model) if not config.use_graph else LavisHypGraphBLIPWithQueue(config, model)
        # if config.use_graph:
            # model = HypGraphCLIPWithQueue(config) if "clip" in config.model_ckt else HypGraphBLIPWithQueue(config)
        # else:
            # model = HypCLIPWithQueue(config) if "clip" in config.model_ckt else HypBLIPWithQueue(config)
        # queue_model = PerceiverLavisBLIPWithQueue(config, model) 
        # distiled_model = DistilLavisBLIP(config, model)
        queue_model = LavisBLIPWithQueue(config, model) if not config.use_graph else LavisHypGraphBLIPWithQueue(config, model)

        trainer = LavisTrainer(
            model=queue_model,
            config=config,
            dataset=coco_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            processor=model.tokenizer,
        )
        # metric = trainer.evaluate(mode='test')
        # print(metric)
        # metric = trainer.evaluate(mode='val')
        # print(metric)
        trainer.train()
    # print(model)
    # inner_training_loop()

    config.epochs = 5 
    config.enable_log = False 
    for curv in [2.0]:
        config.curv = curv
        for use_graph in [False, True]:
            config.use_graph=use_graph
            for manifold in [LORENTZ, EUCLID]:
                config.manifold = manifold 
                # inner_training_loop()
    
