from lavis.datasets.builders import load_dataset
from utils.data_utils import get_fused_dataloader, get_fused_loaders
from model.modules.utils import prepare_processors_and_models
from model.baseFuseModel import BaseModelWithQueue as FuseModel
from trainer_perceiver import MyTrainer as FuseTrainer
from config import parser
from config import (
    EUCLID,
    LORENTZ,
    POINCARE,
    CLIP_LARGE_PATCH_14, 
    CLIP_BASE_PATCH_16, 
    CLIP_BASE_PATCH_32, 
    BLIP_BASE_FLICKR,
    BLIP_BASE_COCO,
    LAVIS_BLIP_BASE_FLICKR, 
    LAVIS_BLIP_BASE_COCO, 
    COCO_PATH, 
    FLICKR_PATH
)

config = parser.parse_args()
def run(config, vis_processors, txt_processors, tokenizers, dataset, models):
    train_loader, val_loader, test_loader = get_fused_loaders(
        dataset,
        vis_processors=vis_processors,
        txt_processors=txt_processors,
        tokenizers=tokenizers,
    )

    queue_model = FuseModel(config, models) 
    trainer = FuseTrainer(
        model=queue_model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    # print(trainer.evaluate('test'))
    # print(trainer.evaluate('val'))
    trainer.train()


if __name__ == '__main__':
    model_ckts = [
        LAVIS_BLIP_BASE_FLICKR, 
        # LAVIS_BLIP_BASE_COCO, 
        # BLIP_BASE_COCO, 
        # CLIP_BASE_PATCH_16, 
        CLIP_LARGE_PATCH_14, 
        # CLIP_BASE_PATCH_32, 
    ]

    if "flickr" in config.dataset:
        config.model_ckt = LAVIS_BLIP_BASE_FLICKR
        dataset = load_dataset("flickr30k", vis_path=FLICKR_PATH, cfg_path=None)
    else:
        config.model_ckt = LAVIS_BLIP_BASE_COCO 
        dataset = load_dataset("coco_retrieval", vis_path=COCO_PATH, cfg_path=None)


    tokenizers, vis_processors, txt_processors, models = prepare_processors_and_models(model_ckts)
    config.epochs = 2 
    config.enable_log = False 
    config.use_margin_loss = False 
    config.curv = 2.0 
    for use_margin_loss  in [False, True]:
        config.use_margin_loss = use_margin_loss 
        for use_fused_feature in [True, False]:
            config.use_fused_features = use_fused_feature 
            for manifold in [LORENTZ, EUCLID]:
                config.manifold = manifold 
                run(config, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, dataset=dataset, models=models)