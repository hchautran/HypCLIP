import argparse

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """
    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)
        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description,
                    )
                else:
                    pass
            else:
                pass
                parser.add_argument(
                    f"--{param}",
                    type=OrNone(default),
                    default=default,
                    help=description,
                )
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


LORENTZ = "lorentz"
EUCLID = "euclidean"
POINCARE = "poincare"
BLIP_BASE_COCO = "Salesforce/blip-itm-base-coco"
BLIP_LARGE_COCO = "Salesforce/blip-itm-large-coco"
CLIP_BASE_PATCH_32 = "openai/clip-vit-base-patch32"
CLIP_BASE_PATCH_16 = "openai/clip-vit-base-patch16"
CLIP_LARGE_PATCH_14 = "openai/clip-vit-large-patch14"
BLIP_BASE = "Salesforce/blip-image-captioning-base"
BLIP_BASE_FLICKR = "Salesforce/blip-itm-base-flickr"
BLIP_LARGE_FLICKR = "Salesforce/blip-itm-large-flickr"
LAVIS_BLIP_BASE_FLICKR = "blip-itm-base-flickr"
LAVIS_BLIP_BASE_COCO= "blip-itm-base-coco"
FLICKR = "nlphuji/flickr30k"
COCO = "coco"

CACHE_DIR = '/mnt/data/.cache'
COCO_PATH = "/mnt/data/itr_dataset/dataset/coco/images"
FLICKR_PATH = "/mnt/data/itr_dataset/dataset/flickr30k/flickr30k_images"

# CACHE_DIR = '/Volumes/ExtraSpace/.cache'
config_args = {
    "training_config": {
        "use_graph":  (False, "use knowledge graph"),
        "lr": (1e-4, "learning rate"),
        "dropout": (0.0, "dropout probability"),
        "cuda": (-1, "which cuda device to use (-1 for cpu training)"),
        "epochs": (10, "maximum number of epochs to train for"),
        "weight_decay": (0.0, "l2 regularization strength"),
        "optimizer": ("adam", "which optimizer to use, can be any of [sgd, adam]"),
        "momentum": (0.995, "momentum in optimizer"),
        "patience": (5, "patience for early stopping"),
        "seed": (42, "seed for training"),
        "log_freq": (1, "how often to compute print train/val metrics (in epochs)"),
        "save": (0, "1 to save model and logs and 0 otherwise"),
        "save_dir": (
            None,
            "path to save training logs and model weights (defaults to logs/task/date/run/)",
        ),
        "sweep_c": (0, ""),
        "lr_reduce_freq": (
            5000,
            "reduce lr every lr-reduce-freq or None to keep lr constant",
        ),
        "gamma": (0.75, "gamma for lr scheduler"),
        "grad_clip": (
            None,
            "max norm for gradient clipping, or None for no gradient clipping",
        ),
        "min_epochs": (2, "do not early stop before min-epochs"),
        "mixed_precision": (
            "fp16",
            "Whether or not to use mixed precision training. Choose from 'no','fp16','bf16' or 'fp8'",
        ),
        "gradient_accumulation_steps": (
            1,
            "The number of steps that should pass before gradients are accumulated",
        ),
        "lorentz_pos_margin": (
            0.0,
            "decision margin for hyperbolic maninfold (0.0 for no margin)",
        ),
        "lorentz_neg_margin": (
            0.8,
            "decision margin for hyperbolic manifold (0.0 for no margin)",
        ),
        "euclid_pos_margin": (
            1.0,
            "decision margin for euclid manifold (0.0 for no margin)",
        ),
        "euclid_neg_margin": (
            0.5,
            "decision margin for euclid manifold (0.0 for no margin)",
        ),
        "max_txt_len": (35, "max_txt_len"),
        "negative_all_rank": (False, "negative_all_rank"),
        "alpha": (0.4, "alpha"),
        "queue_size": (64000, "queue size"),
        "batch_size": (40, "batch size"),
        "eval_freq": (1200, "how often to compute val metrics (in epochs)"),
        "weight_i2t": (0.5, "weight image to text"),
        "enable_log": (True, "enable log"),
        "use_margin_loss": (True, "use margin loss"),
        "use_graph_loss": (False, "use margin loss for graph"),
        "use_entailment_loss": (False, "use entailment loss"),
        "hyp_margin_loss_weight": (0.0, "hyperbolic margin loss weight"),
        "num_proj_layers": (6, "number of project layers"),
        "proj_layer_hidden_sizes": (512, "hidden size of proj layers"),
        "normalize_text_embed": (False,""),
        "normalize_image_embed": (False,""),
        "shared_proj_layers": (False, "number of project layers"),
        "use_itm_head": (True, "use itm head"),
        "use_root": (False, "use graph root"),
        "graph_hidden_channels": (512, "graph size"),
    },
    "hybrid_model_config": {
        "model_ckt": (CLIP_BASE_PATCH_16, "model checkpoint on Hugging Face"),
        "manifold": (
            EUCLID,
            "which manifold to use [euclidean, lorentz]",
        ),
        "curv": (1.0, "hyperbolic curvature"),
        "atol": (1e-1, "The relative tolerance parameter"),
        "rtol": (1e-1, "The absolute tolerance parameter"),
        "temp": (0.05, "distance temperature"),
        "clip_radius": (1.25, "clipping radius"),
        "vision_trainable_blocks": (3, "number of trainable blocks in vision model"),
        "text_trainable_blocks": (12, "number of trainable blocks in text model"),
        "num_vision_hidden_states": (1, "number of trainable blocks in vision model"),
        "num_text_hidden_states": (1, "number of trainable blocks in text model"),
        "ft_out": (512, "final project dimension"),
        "curv_learnable": (False, "is curvature learnable"),
        "freeze_embedding": (True, "freeze embedding layers"),
        "use_lorentz_centroid": (False, "use lorentz centroid pooler"),
        "fourier": (False, "fourier"),
        "soft_target_loss": (0.4, "soft_target_loss"),
    },
    "data_config": {
        "dataset": (FLICKR, "which dataset to use"),
        "cache_dir": (CACHE_DIR, "cache_dir"),
    },
    "perceiver": {
        "num_latents": (32, "which dataset to use"),
        "d_latents": (512, "d latent"),
        "num_blocks": (1, "d out"),
        "num_self_attends_per_block": (6, "cache_dir"),
        "num_cross_attention_heads": (2, "cache_dir"),
        "num_self_attention_heads": (2, "cache_dir"),
        "cross_attention_widening_factor": (2, "cache_dir"),
        "attention_probs_dropout_prob": (0.4, "cache_dir"),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
