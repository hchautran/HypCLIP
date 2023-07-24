import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Flickr_dataset(Dataset):
    def __init__(self, config, dataset, processor ):  
        self.device = torch.device(config['device'])
        self.dataset = dataset
        self.processor = processor
        self.dataset_len = len(dataset)
        self.cap_per_img = 5
            
    def __len__(self):
        return len(self.dataset)  * self.cap_per_img
    
    def __getitem__(self, index): 
        data = self.dataset[int(index / self.cap_per_img)]
        out = {
            'img_id': data['img_id'],
            'input_ids': data['input_ids'][int(index % self.cap_per_img)],
            'attention_mask': data['attention_mask'][(index % self.cap_per_img)],
            'pixel_values': data['pixel_values'],
            'image': data['image'],
            'caption': data['caption'][int(index % self.cap_per_img)],
            'sent_id': data['sentids'][int(index % self.cap_per_img)]
        }
        return out

        
def get_dataloader(dataset:Flickr_dataset, batch_size, processor):

    def collate_func(batch, processor):
        df = pd.DataFrame(batch)
        return processor(
            text=list(df['caption']), 
            images=list(df['image']),
            padding=True, 
            return_tensors='pt'
        )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn = lambda batch: collate_func(batch, processor),
        shuffle=True
    ) 
