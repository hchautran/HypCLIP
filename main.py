import torch
from datasets import load_dataset
from typing_extensions import Literal
from transformers import CLIPVisionModel, CLIPTextModel


if __name__ == '__main__':
    from config import parser
    args = parser.parse_args()
    vision_model = CLIPVisionModel.from_pretrained(args.model_ckt) 
    text_model = CLIPTextModel.from_pretrained(args.model_ckt) 
    print(vision_model.num_parameters())
    print(text_model.num_parameters())
    
    
