import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
import numpy as np
from transformers import CLIPProcessor 
from datasets import dataset_dict 
from datasets import load_dataset
from lavis.datasets.builders import load_dataset as lavis_dataset
class Flickr_dataset(Dataset):
    def __init__(self, dataset, load_raw_image=False):  
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.cap_per_img = 5
        self.load_raw_image=True

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



class CoFlickr_dataset(Dataset):
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
            'clip_pixel_values': data['clip_pixel_values'],
            'blip_pixel_values': data['blip_pixel_values'],
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
    data['img_id'] = torch.tensor(list(df['img_id'])) 
    return data





def co_collate_func(batch, clip_processor, blip_processor):
    # print(batch)
    df = pd.DataFrame(batch)
    data = {} 
    clip_data = clip_processor(
        text=list(df['caption']), 
        padding=True, 
        return_tensors='pt',
        truncation=True
    )
    blip_data = blip_processor(
        text=list(df['caption']), 
        padding=True, 
        return_tensors='pt',
        truncation=True
    )
    data['clip_input_ids'] = clip_data['input_ids'] 
    data['clip_attention_mask'] = clip_data['attention_mask'] 
    data['blip_input_ids'] = blip_data['input_ids'] 
    data['blip_attention_mask'] = blip_data['attention_mask'] 
    data['clip_pixel_values'] = torch.from_numpy(np.concatenate(list(df['clip_pixel_values']), 0))
    data['blip_pixel_values'] = torch.from_numpy(np.concatenate(list(df['blip_pixel_values']), 0))
    data['img_id'] = torch.tensor(list(df['img_id'])) 
    return data

def get_coco_dataloader(dataset, batch_size, vis_processor, txt_processor, mode='train'):
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco_images"
    def coco_collate_func(batch, processor):
    # print(batch)
        data = {}
        data['pixel_values']= vis_processor(batch['image'])
        data['img_id'] = batch['image_id']
        data['input_ids'] = batch['image_id']
        data['attention_mask'] = txt_processor(batch['text_input'])
        return data

    coco_dataset = get_coco(cache_dir=COCO_PATH) 
    custom_sampler = UniqueClassSampler(coco_dataset, batch_size)
    if mode == 'train':
        return DataLoader(
            coco_dataset, 
            batch_size=batch_size, 
            collate_fn = lambda batch: collate_func(batch, processor),
            shuffle=True
        )

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

def get_dataloader(dataset, batch_size, processor, mode='train', use_random_sampler=False):
    flickr_dataset = Flickr_dataset(dataset) 
    custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)
    if mode == 'train':
        if use_random_sampler:
            return DataLoader(
                flickr_dataset, 
                batch_size=batch_size, 
                collate_fn = lambda batch: collate_func(batch, processor),
                shuffle=True
            )

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

def get_co_dataloader(dataset, batch_size, clip_processor, blip_processor,mode='train', use_random_sampler=False):
    flickr_dataset = CoFlickr_dataset(dataset) 
    custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)
    if mode == 'train':
        if use_random_sampler:
            return DataLoader(
                flickr_dataset, 
                batch_size=batch_size, 
                collate_fn = lambda batch: co_collate_func(batch, clip_processor, blip_processor),
                shuffle=True
            )
 
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn = lambda batch: co_collate_func(batch, clip_processor, blip_processor),
            sampler=custom_sampler,
        ) 
    else:
        return DataLoader(
            flickr_dataset, 
            batch_size=batch_size, 
            collate_fn = lambda batch: co_collate_func(batch, clip_processor, blip_processor),
            shuffle=False
        )
        

def preprocess_img(sample, processor:CLIPProcessor):
    sample['pixel_values'] = processor(images=sample['image'])['pixel_values']
    return sample

def lavis_preprocess_img(sample, processor):
    sample['pixel_values'] = processor(sample['image']).unsqueeze(0)
    return sample


def co_preprocess_img(sample, clip_processor, blip_processor):
    sample['clip_pixel_values'] = clip_processor(images=sample['image'])['pixel_values']
    sample['blip_pixel_values'] = blip_processor(sample['image']).unsqueeze(0)
    return sample

def parse_int(sample):
    sample['img_id'] = int(sample['img_id'])
    return sample
    
def get_flickr(flickr_ckt, cache_dir):
    flickr30k = load_dataset(flickr_ckt, cache_dir=cache_dir).remove_columns(['sentids', 'filename']).map(parse_int)
    ds = dataset_dict.DatasetDict({
        'train' : flickr30k.filter(lambda x: x['split'] == 'train')['test'], 
        'test' : flickr30k.filter(lambda x: x['split'] == 'test')['test'], 
        'val' : flickr30k.filter(lambda x: x['split'] == 'val')['test'], 
    }) 
    return ds


def get_coco(cache_dir):
    coco_dataset = lavis_dataset("coco_caption", vis_path=cache_dir)
    return coco_dataset