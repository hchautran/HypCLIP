import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
from datasets import load_dataset
from model.hypCLIP import HypCLIP
from datasets import load_dataset 
from transformers import CLIPProcessor
from tqdm.auto import tqdm
from utils.data_utils import get_dataloader
from trainer import HypCLIPTrainer





if __name__ == '__main__':
    from config import parser
    config = parser.parse_args()


    processor = CLIPProcessor.from_pretrained(config.model_ckt)
    # flickr30k = load_dataset(config.dataset).with_format('numpy')

    # train_loader = get_dataloader(flickr30k['train'], config.batch_size, processor=processor)
    # val_loader = get_dataloader(flickr30k['val'], 5, processor=processor, mode='val')
    # test_loader = get_dataloader(flickr30k['test'], 5, processor=processor, mode='test')

    # model = HypCLIP(config) 
    # # print(vars(config))
    # print('number of params', model.num_parameters())
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainer = HypCLIPTrainer(config=config, processor=processor)
    # trainer.evaluate()
    trainer.train()
    


        
