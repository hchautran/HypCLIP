import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
import time
import numpy as np
from transformers import CLIPProcessor 
from lavis.models import load_model_and_preprocess

class Flickr_dataset(Dataset):
    def __init__(self, dataset):  
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.cap_per_img = 5

    def __len__(self):
        return self.dataset_len  * self.cap_per_img
    
    def __getitem__(self, index): 
        data = self.dataset[int(index / self.cap_per_img)]
        out = {
            'img_id': data['img_id'],
            'pixel_values': data['pixel_values'],
            'caption': data['caption'][int(index % self.cap_per_img)],
        }
        return out

class UniqueClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.classes= {}
        self.remain_sample = 0

        progress = tqdm(range(len(data_source.dataset['img_id'])))

        for index, img_id in enumerate(data_source.dataset['img_id']):
            self.classes[img_id] = [data_source.cap_per_img * index + i for i in range(data_source.cap_per_img)]
            self.remain_sample += 5
            progress.update(1)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        while self.remain_sample >= self.batch_size:
            if len(self.classes.keys()) >= self.batch_size:
              batch_classes = np.random.choice(list(self.classes.keys()), self.batch_size, replace=False)
            else:
              batch_classes = list(self.classes.keys())
            for img_id in batch_classes:
                indices = self.classes[img_id] 
                selected_index = np.random.choice(indices)
                yield selected_index
                self.classes[img_id] = np.setdiff1d(self.classes[img_id], selected_index)
                self.remain_sample -=1
                if len(self.classes[img_id]) == 0:
                    self.classes.pop(img_id)


def collate_func(batch, processor):
    # print(batch)
    df = pd.DataFrame(batch)
    data = processor(
        text=list(df['caption']), 
        padding=True, 
        return_tensors='pt',
        truncation=True
    )
    data['pixel_values'] = torch.from_numpy(np.concatenate(list(df['pixel_values']), 0))
    data['image_id'] = torch.tensor(list(df['img_id'])) 
    return data


def collate_func_lavis(batch):
    # print(batch)
    df = pd.DataFrame(batch)
    sample = {
        'image':  torch.from_numpy(np.concatenate(list(df['pixel_values']), 0)),
        'text_input': list(df['caption']),
        'image_id': torch.tensor(list(df['img_id'])),
    
    }
    return sample
        

def get_dataloader(dataset, batch_size, processor, mode='train'):
    flickr_dataset = Flickr_dataset(dataset) 
    custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)
    if mode == 'train':
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn = lambda batch: collate_func(batch, processor),
            sampler=custom_sampler,
        ) 
    else:
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn = lambda batch: collate_func(batch, processor),
            shuffle=False
        )

        
def get_lavis_dataloader(dataset, batch_size, mode='train'):
    flickr_dataset = Flickr_dataset(dataset) 
    custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)
    if mode == 'train':
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn = collate_func_lavis,
            sampler=custom_sampler,
        ) 
    else:
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn =collate_func_lavis, 
            shuffle=False
        )

def preprocess_img(sample, processor:CLIPProcessor):
    sample['pixel_values'] = processor(images=sample['image'])['pixel_values']
    return sample

def preprocess_img_lavis(sample, lavis_vis_processor):
    sample['pixel_values'] = lavis_vis_processor['eval'](sample['image']).unsqueeze(0)
    return sample

def parse_int(sample):
    sample['img_id'] = int(sample['img_id'])
    return sample
    

    


if __name__ == '__main__':
    from datasets import load_dataset
    from tqdm.auto import tqdm
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    flickr30k = load_dataset('EddieChen372/flickr')
    model, vis_processor, text_processor = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    flickr30k = flickr30k.map(lambda sample: preprocess_img_lavis(sample, lavis_vis_processor=vis_processor), num_proc=4).remove_columns(['image'])
    # # flickr30k = flickr30k.map(lambda sample: preprocess_img(sample, processor)).remove_columns(['image'])
    # flickr30k.set_format('numpy')
    # print(flickr30k)
    
    # # print(model)
    # batch_size=200
    # # print(flickr30k)
    # train_loader = get_lavis_dataloader(flickr30k['train'], batch_size=batch_size, mode='train')
    # # train_loader = get_dataloader(
    # #     flickr30k['train'], 
    # #     batch_size=batch_size, 
    # #     mode='train', 
    # #     processor=processor
    # # )



    
    
    