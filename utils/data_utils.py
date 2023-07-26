import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
import time
import numpy as np

class Flickr_dataset(Dataset):
    def __init__(self, dataset):  
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.data = self.dataset.to_pandas()
        self.cap_per_img = 5

    def __len__(self):
        return len(self.dataset)  * self.cap_per_img
    
    def __getitem__(self, index): 
        data = self.data.iloc[int(index / self.cap_per_img)]
        out = {
            'img_id': data['img_id'],
            'pixel_values': data['pixel_values'],
            'caption': data['caption'][int(index % self.cap_per_img)],
            'sent_id': data['sentids'][int(index % self.cap_per_img)]
        }
        return out

class UniqueClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        print('Preparing dataloader...')
        self.data_source = data_source
        self.batch_size = batch_size
        self.classes= {}
        self.remain_sample = 0
        # print(data_source.dataset['img_id'])

        progress = tqdm(range(len(data_source.dataset['img_id'])))

        for index, img_id in enumerate(data_source.dataset['img_id']):
            self.classes[img_id] = [ data_source.cap_per_img * index + i for i in range(data_source.cap_per_img)]
            self.remain_sample += 5
            progress.update(1)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        start = time.time()
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
        print(time.time() - start)  




def collate_func(batch, processor):
    # print(batch)
    df = pd.DataFrame(batch)
    data = processor(
        text=list(df['caption']), 
        # images=list(df['image']),
        padding=True, 
        return_tensors='pt'
    )
    return df['img_id'], data 
        

def get_dataloader(dataset, batch_size, processor):

    flickr_dataset = Flickr_dataset(dataset) 
    custom_sampler = UniqueClassSampler(flickr_dataset, batch_size)

    return DataLoader(
        flickr_dataset, 
        batch_size=batch_size, 
        collate_fn = lambda batch: collate_func(batch, processor),
        sampler=custom_sampler,
        # num_workers=2
        # shuffle=True
    ) 


if __name__ == '__main__':
    from datasets import load_dataset 
    from transformers import CLIPProcessor
    from tqdm.auto import tqdm
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    flickr30k = load_dataset('EddieChen372/flickr30k',split='val' ,keep_in_memory=True)
    batch_size = 128 

    # train_loader = get_dataloader(flickr30k['train'], batch_size, processor=processor)
    val_loader = get_dataloader(flickr30k, batch_size, processor=processor)
    # test_loader = get_dataloader(flickr30k['test'], batch_size, processor=processor)


    for img_ids, data in tqdm(val_loader):
        pass
        print(list(img_ids))
        # print(data['input_ids'].shape)
        # print(data['attention_mask'].shape)
        # print(data['pixel_values'].shape)
        
    
    
    
    