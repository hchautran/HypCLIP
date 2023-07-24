import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
from datasets import load_dataset


if __name__ == '__main__':
    from config import parser
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    
    vision_model = CLIPVisionModel.from_pretrained(args.model_ckt, cache_dir=args.cache_dir) 
    text_model = CLIPTextModel.from_pretrained(args.model_ckt, cache_dir=args.cache_dir) 
    processor = CLIPProcessor.from_pretrained(args.model_ckt, cache_dir=args.cache_dir)
    print(vision_model.num_parameters())
    print(text_model.num_parameters())
    
    
