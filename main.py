import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor, BlipForImageTextRetrieval
from datasets import load_dataset
from model.hypCLIP import HypCLIP
from model.hypBLIP import HypBLIP 
from datasets import load_dataset 
from transformers import CLIPProcessor, BlipProcessor
from tqdm.auto import tqdm
from utils.data_utils import get_dataloader, preprocess_img
from trainer import MyTrainer
from trainer_with_queue import MyTrainerWithMomentum 
from accelerate import find_executable_batch_size
from utils.data_utils import get_flickr



if __name__ == '__main__':
    from config import parser
    from config import EUCLID, LORENTZ, POINCARE
    config = parser.parse_args()
    if 'blip' in config.model_ckt:
        print("Getting BLIP processor...")
        processor = BlipProcessor.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
    else:
        print("Getting CLIP processor...")
        processor = CLIPProcessor.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)

    if 'flickr' in config.dataset:
        dataset = get_flickr(config.dataset, cache_dir=config.cache_dir) 
    else:
        dataset = get_flickr(config.dataset, cache_dir=config.cache_dir) 

    dataset = (
        dataset
        .map(lambda sample: preprocess_img(sample, processor=processor))
        .remove_columns(['image'])
    )
    dataset.set_format('numpy')




    @find_executable_batch_size(starting_batch_size=config.batch_size)
    def inner_training_loop(batch_size):
        config.batch_size=batch_size
        train_loader = get_dataloader(dataset['train'], config.batch_size, processor=processor, mode='train', use_random_sampler=True)
        test_loader = get_dataloader(dataset['test'], 5, processor=processor, mode='test')
        val_loader = get_dataloader(dataset['val'], 5, processor=processor, mode='val')
        model = HypCLIP(config) if 'clip' in config.model_ckt  else HypBLIP(config)
        trainer = MyTrainerWithMomentum(
            model=model, 
            config=config, 
            dataset=dataset ,
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader,
            processor=processor
        )
        # print(trainer.evaluate(mode='test'))
        trainer.train()
        print(trainer.evaluate(mode='test'))

    for manifold in [LORENTZ]:
        config.manifold = manifold 
        inner_training_loop()

    
    
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-flickr", cache_dir=config.cache_dir)
    # model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-flickr", cache_dir=config.cache_dir, torch_dtype=torch.float16).to("cuda")

    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # question = "A woman and a dog sitting together in a beach."
    # inputs = processor(raw_image, questnmduy2020ion, return_tensors="pt").to("cuda", torch.float16)

    # itm_scores = model(**inputs)[0]
    # cosine_score = model(**inputs, use_itm_head=False)[0] 


        
