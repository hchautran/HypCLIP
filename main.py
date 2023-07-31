import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
from datasets import load_dataset
from model.hypCLIP import HypCLIP
from datasets import load_dataset 
from transformers import CLIPProcessor
from tqdm.auto import tqdm
from utils.data_utils import get_dataloader
from trainer import HypCLIPTrainer
from accelerate import find_executable_batch_size



if __name__ == '__main__':
    from config import parser
    config = parser.parse_args()
    
    # for batch_size in [100,150,200]:    
    #     config.batch_size=batch_size
        # for curv in [0.05,0.1]:
            # config.curvature = curv
    #     for manifold in ['euclidean', 'lorentz']:
    #         config.manifold = manifold
    #         for ft_out in [128,256,512,1024]:
    #             config.ft_out = ft_out 


    # processor = CLIPProcessor.from_pretrained(config.model_ckt, cache_dir=config.cache_dir)
    # flickr30k = load_dataset('EddieChen372/flickr30k', cache_dir=config.cache_dir).with_format('numpy')
    # train_loader = get_dataloader(flickr30k['train'], config.batch_size, processor=processor)
    # val_loader = get_dataloader(flickr30k['val'], 5, processor=processor, mode='val')
    # test_loader = get_dataloader(flickr30k['test'], 5, processor=processor, mode='test')

    # model = HypCLIP(config) 
    # # print(vars(config))
    # print('number of params', model.num_parameters())
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # for img_ids, batch in tqdm(train_loader):
    #     assert len(img_ids) == len(set(img_ids))
    # for img_ids, batch in tqdm(test_loader):
    #     assert len(set(img_ids)) == 1 
    # for img_ids, batch in tqdm(val_loader):
    #     assert len(set(img_ids)) == 1 
    @find_executable_batch_size(starting_batch_size=config.batch_size)
    def inner_training_loop(batch_size):
        config.batch_size=batch_size
        trainer = HypCLIPTrainer(config=config)
        trainer.train()

    inner_training_loop()

    
    


        
