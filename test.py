import torch
from model.hypCLIP import HypGraphCLIPWithQueue, HypCLIPWithQueue 
from model.hypBLIP import HypBLIPWithQueue, HypGraphBLIPWithQueue

from lavis import CLIP
from model.hypBLIP import LavisBLIPWithQueue ,LavisHypGraphBLIPWithQueue
from lavis.datasets.builders import load_dataset
from trainer_queue import MyTrainer as LavisTrainer 
from utils.data_utils import  get_loaders
from tqdm.auto import tqdm

from lavis.models import load_model_and_preprocess

if __name__ == "__main__":
    from config import parser
    from config import EUCLID, LORENTZ, LAVIS_BLIP_BASE_COCO, LAVIS_BLIP_BASE_FLICKR
    COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
    FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"

    config = parser.parse_args()
    model, vis_processors, txt_processors = load_model_and_preprocess("blip2", "coco", is_eval=True)
    print(model)
  