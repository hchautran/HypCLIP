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
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'
BLIP_BASE_FLICKR = 'Salesforce/blip-itm-base-flickr' 
BLIP_LARGE_FLICKR = 'Salesforce/blip-itm-large-flickr' 
BLIP_BASE_COCO = 'Salesforce/blip-itm-base-coco' 
BLIP_LARGE_COCO = 'Salesforce/blip-itm-large-coco' 
CLIP_BASE_PATCH_32 = 'openai/clip-vit-base-patch32'
CLIP_BASE_PATCH_16 = 'openai/clip-vit-base-patch16'
CLIP_LARGE_PATCH_14  = 'openai/clip-vit-large-patch14'
FLICKR = 'nlphuji/flickr30k' 
CACHE_DIR = '/Volumes/ExtraSpace/.cache' 
# CACHE_DIR = '/mnt/data/.cache'

config_args = {
    'training_config': {
        'lr': (1e-4, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (100, 'maximum number of epochs to train for'),
        'weight_decay': (0., 'l2 regularization strength'),
        'optimizer': ('adam', 'which optimizer to use, can be any of [sgd, adam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (25, 'patience for early stopping'),
        'seed': (42, 'seed for training'),
        'log_freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval_freq': (-1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save_dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep_c': (0, ''),
        'lr_reduce_freq': (5000, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'grad_clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min_epochs': (25,'do not early stop before min-epochs'),
        'batch_size': (60,'batch size'),
        'enable_log': (False, 'enable log'),
        'mixed_precision': ("fp16", "Whether or not to use mixed precision training. Choose from 'no','fp16','bf16' or 'fp8'"),
        'gradient_accumulation_steps': (1, "The number of steps that should pass before gradients are accumulated"),
        'lorentz_pos_margin': (1.0, 'decision margin for hyperbolic maninfold (0.0 for no margin)'),
        'lorentz_neg_margin': (3.0, 'decision margin for hyperbolic manifold (0.0 for no margin)'),
        'euclid_pos_margin': (1.0, 'decision margin for euclid manifold (0.0 for no margin)'),
        'euclid_neg_margin': (0.25, 'decision margin for euclid manifold (0.0 for no margin)'),
        'alpha': (0.4, 'alpha'),
        'queue_size': (64000, 'queue_size'),
    },
    'model_config': {
        'model_ckt':(CLIP_BASE_PATCH_16, 'model checkpoin on Hugging Face'),
        'manifold': (EUCLID, 'which manifold to use, can be any of [euclidean, poincare, lorentz]'),
        'curv': (0.05, 'hyperbolic curvature'),
        'temp': (0.07, 'distance temperature'),
        'clip_radius': (2.5, 'clipping radius'),
        'vision_trainable_blocks': (9, 'number of trainable blocks in vision model'),
        'text_trainable_blocks': (9, 'number of trainable blocks in text model'),
        'ft_out': (128, 'final project dimension'),
        'curv_learnable': (False, 'is curvature learnable') ,
        'freeze_embedding': (False, 'freeze embedding layers')
    },
    'data_config': {
        'dataset': (FLICKR, 'which dataset to use'),
        'cache_dir': (CACHE_DIR, 'cache_dir'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
