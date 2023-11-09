import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
import numpy as np
from transformers import CLIPProcessor, BlipProcessor 
from datasets import dataset_dict 
from datasets import load_dataset
from lavis.datasets.builders import load_dataset as lavis_dataset
class Flickr_dataset(Dataset):
    def __init__(self, dataset):  
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

class EvalDataset(Dataset):
    def __init__(self, dataset, vis_processor, tokenizer, txt_processor=None):  
        self.dataset = dataset
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor 
        self.tokenizer = tokenizer 
        self.img2txt = dataset.img2txt
        self.txt2img = dataset.txt2img
        self.text = dataset.text
        self.image = dataset.image

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index): 
        data =  self.dataset[index]
        cap_indexes = self.dataset.img2txt[index]
        captions = []
        output = {}
        for i in cap_indexes:
            if self.txt_processor is not None:
                captions.append(self.txt_processor(self.text[i]))
            else:
                captions.append(self.text[i])

        if isinstance(self.tokenizer, CLIPProcessor) or isinstance(self.tokenizer, BlipProcessor):
            text_inputs = self.tokenizer(text=captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
            output['pixel_values'] = self.vis_processor(images=data['image'], return_tensors='pt')['pixel_values']
        else:
            text_inputs = self.tokenizer(captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
            output['pixel_values'] = self.vis_processor(data['image']).unsqueeze_(0)


        output['img_id'] = data['index']
        output['input_ids'] = text_inputs['input_ids']
        output['attention_mask'] = text_inputs['attention_mask']
        return output


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

def get_dataloader(dataset,  vis_processor, tokenizer, txt_processor=None, mode='train', batch_size=1):
    def coco_collate_func(batch):
        df = pd.DataFrame(batch)
        data = {}
        pixel_values = []
        texts = []

        for i, image in enumerate(list(df['image'])):
            if isinstance(vis_processor, CLIPProcessor) or isinstance(vis_processor, BlipProcessor):
                pixel_values.append(vis_processor(images=image, return_tensors='pt')['pixel_values'])
            else:
                pixel_values.append(vis_processor(image).unsqueeze_(0))
            if txt_processor is not None:
                texts.append(txt_processor(list(df['text_input'])[i]))
            else:
                texts.append(list(df['text_input'])[i])
        if isinstance(tokenizer, CLIPProcessor) or isinstance(tokenizer, BlipProcessor):
            text_inputs = tokenizer(text=texts, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
        else:
            text_inputs = tokenizer(texts, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
        data['pixel_values'] = torch.cat(pixel_values, dim=0)
        data['img_id'] = torch.tensor(list(df['image_id']))
        data['input_ids'] = text_inputs['input_ids']
        data['attention_mask'] = text_inputs['attention_mask']
        return data

    if mode == 'train':
        return DataLoader(
            dataset[mode], 
            batch_size=batch_size, 
            collate_fn=coco_collate_func,
            shuffle=True
        ) 
    else:
        cur_dataset = EvalDataset(
            dataset[mode], 
            vis_processor=vis_processor,
            txt_processor=txt_processor,
            tokenizer=tokenizer
        )
        return cur_dataset

# def get_dataloader(dataset, batch_size, processor, mode='train', use_random_sampler=False):
#     flickr_dataset = Flickr_dataset(dataset) 
#     custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)
#     if mode == 'train':
#         if use_random_sampler:
#             return DataLoader(
#                 flickr_dataset, 
#                 batch_size=batch_size, 
#                 collate_fn = lambda batch: collate_func(batch, processor),
#                 shuffle=True
#             )

#         return DataLoader(
#             flickr_dataset, 
#             batch_size=batch_size, 
#             collate_fn = lambda batch: collate_func(batch, processor),
#             sampler=custom_sampler,
#         ) 
#     else:
#         return DataLoader(
#             flickr_dataset, 
#             batch_size=batch_size, 
#             collate_fn = lambda batch: collate_func(batch, processor),
#             shuffle=False
#         )

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


def get_coco(vis_processor, txt_processor):
    from lavis.datasets.datasets.retrieval_datasets import RetrievalDataset
    coco_dataset = {}
    coco_dataset = RetrievalDataset(
        vis_processor=vis_processor,
        text_processor=txt_processor,
        ann_paths=[
            '/mnt/data/itr_dataset/dataset/coco/annotations/coco_karpathy_train.json',
            '/mnt/data/itr_dataset/dataset/coco/annotations/coco_karpathy_test.json',
            '/mnt/data/itr_dataset/dataset/coco/annotations/coco_karpathy_val.json',
        ],
        vis_root='/mnt/data/itr_dataset/dataset/coco/images/',
    )

    return coco_dataset

def get_loaders(config, dataset, vis_processor, tokenizer, txt_processor=None):
    train_loader  = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        vis_processor=vis_processor,
        txt_processor=txt_processor,
        tokenizer=tokenizer,
        mode='train',
    ) 
    test_loader  = get_dataloader(
        dataset=dataset,
        vis_processor=vis_processor,
        txt_processor=txt_processor,
        tokenizer=tokenizer,
        mode='test',
    ) 
    val_loader  = get_dataloader(
        dataset=dataset,
        vis_processor=vis_processor,
        txt_processor=txt_processor,
        tokenizer=tokenizer,
        mode='val',
    ) 
    return train_loader, val_loader, test_loader
