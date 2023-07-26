import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
from datasets import load_dataset
from model.hypCLIP import HypCLIP
from datasets import load_dataset 
from transformers import CLIPProcessor
from tqdm.auto import tqdm
from utils.data_utils import get_dataloader


if __name__ == '__main__':
    from config import parser
    config = parser.parse_args()


    processor = CLIPProcessor.from_pretrained(config.model_ckt)
    flickr30k = load_dataset(config.dataset)

    train_loader = get_dataloader(flickr30k['train'], config.batch_size, processor=processor)
    val_loader = get_dataloader(flickr30k['val'], config.batch_size, processor=processor)
    test_loader = get_dataloader(flickr30k['test'], config.batch_size, processor=processor)

    model = HypCLIP(config) 
    print('number of params', model.num_parameters())

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            print(data['input_ids'].shape)
            print(data['attention_mask'].shape)
            print(data['pixel_values'].shape)
            output = model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                pixel_values=data['pixel_values'],
            )
            print(output['text_embeds'].shape)
            print(output['image_embeds'].shape)

            print(output['sim_per_image'].shapes)
            print(output['sim_per_text'].shape)


        
