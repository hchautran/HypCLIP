
import datasets
from torch.utils.data import DataLoader
from transformers import CLIPProcessor


    


def get_data_loader(dataset, data_collator,  processor_ckt, batch_size=8,  ):
    processor = CLIPProcessor.from_pretrained(processor_ckt)
    def process_data(sample):
        output = processor(text=sample['caption'], images=sample['image'], return_tensors="pt", padding=True)
        sample['']

    dataset = dataset.map(process_data)
    
    
    data_loader = DataLoader(
        dataset = dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
        

        
if __name__ == '__main__':
    from datasets import load_dataset
    processor_ckt= 'openai/clip-vit-base-patch32'
    dataset_name = 'nlphuji/flickr30k'
    processor = CLIPProcessor.from_pretrained(processor_ckt)

    ds = load_dataset(dataset_name)
    train_set = ds['test'].filter(lambda x: x['split']=='train')
    test_set = ds['test'].filter(lambda x: x['split']=='test')
    val_set= ds['test'].filter(lambda x: x['split']=='val')

    print(train_set)

    
    
        
    