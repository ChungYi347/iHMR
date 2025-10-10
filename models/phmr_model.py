import torch
import argparse
from yacs.config import CfgNode as CN
from .matcher import build_matcher
from .criterion import SetCriterionHMR


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 4
cfg.SEED_VALUE = -1
cfg.IMG_RES = 224
cfg.PIN_MEMORY = False

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.RESUME = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.IS_FINETUNE = False
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 100
cfg.TRAIN.WARMUP_STEPS = 1
cfg.TRAIN.LR_SCHEDULE = []
cfg.TRAIN.LR_DECAY = 1.0
cfg.TRAIN.MAX_STEP = 100000
cfg.TRAIN.CLIP_GRADIENT = 1.0
cfg.TRAIN.WD = 0.01
cfg.TRAIN.OPT = 'AdamW'
cfg.TRAIN.GRAD_ACC = 1
cfg.TRAIN.FINETUNE = None

cfg.DATASET = CN()
cfg.DATASET.TEST = 'emdb_1'
cfg.DATASET.SEQ_LEN = 16
cfg.DATASET.STRIDE = 16
cfg.DATASET.RESCALE_TO_BEDLAM = 0.25

cfg.MODEL = CN()
cfg.MODEL.PRETRAINED = None
cfg.MODEL.REG_LAYER = 2
cfg.MODEL.DROP_PATH = 0.3
cfg.MODEL.MOTION_MODULE = False
cfg.MODEL.MOTION_DIM = 384
cfg.MODEL.MLP_DIM=2048
cfg.MODEL.TRANSFORMER = 'oneway'
cfg.MODEL.INVERSE_DEPTH = False
cfg.MODEL.USE_PROMPT = True
cfg.MODEL.TEXT_PROMPT = 'auxiliary'
cfg.MODEL.TEXT_ENCODER = 'CLIP'
cfg.MODEL.CLIP_DROPOUT = 0.0
cfg.MODEL.PROMPT_TYPE = 'prompt'
cfg.MODEL.CAM_ENCODER = 'ray'
cfg.MODEL.TRANSL = 'transl'
cfg.MODEL.MASK_PROMPT = False

cfg.LOSS = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    cfg.cfg_file = cfg_file
    return cfg.clone()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # cfg_file = args.cfg
    cfg_file = 'weights/pretrain/phmr/config.yaml'
    if cfg_file is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()

    return cfg


def load_model_from_folder(folder):
    from models.prompt_model import build_phmr
    cfg = f'{folder}/config.yaml'
    cfg = parse_args(['--cfg', cfg])

    ckpt = f'{folder}/checkpoint.ckpt'
    weight = torch.load(ckpt, map_location='cuda', weights_only=True)

    model = build_phmr(cfg)
    model = model.cuda()

    skip_prefixes = ["prompt_encoder.mask_downscaling.", "prompt_encoder.clip_encoder.", 
        "smpl_decoder.transformer.layers.0.cross_person_attn.", 
        "smpl_decoder.transformer.layers.1.cross_person_attn.", 
        "smpl_decoder.transformer.layers.2.cross_person_attn.", 
        "smpl_decoder.transformer.layers.3.cross_person_attn.", 
        # "smpl_decoder.transformer.layers.0.norm2.weight", "smpl_decoder.transformer.layers.0.norm2.bias",
        # "smpl_decoder.transformer.layers.1.norm2.weight", "smpl_decoder.transformer.layers.1.norm2.bias",
        # "smpl_decoder.transformer.layers.2.norm2.weight", "smpl_decoder.transformer.layers.2.norm2.bias",
        # "smpl_decoder.transformer.layers.3.norm2.weight", "smpl_decoder.transformer.layers.3.norm2.bias",
        "prompt_encoder.no_mask_embed.weight", 
        "smpl_decoder.transformer.layers.0.norm5.weight", "smpl_decoder.transformer.layers.0.norm5.bias",
         "smpl_decoder.transformer.layers.0.pe.pos_table", 
         "smpl_decoder.transformer.layers.1.norm5.weight", "smpl_decoder.transformer.layers.1.norm5.bias", 
         "smpl_decoder.transformer.layers.1.pe.pos_table", 
         "smpl_decoder.transformer.layers.2.norm5.weight", "smpl_decoder.transformer.layers.2.norm5.bias", 
         "smpl_decoder.transformer.layers.2.pe.pos_table", 
         "smpl_decoder.transformer.layers.3.norm5.weight", "smpl_decoder.transformer.layers.3.norm5.bias", 
         "smpl_decoder.transformer.layers.3.pe.pos_table"
        ]
    filtered = {k: v for k, v in weight['state_dict'].items()
                if not any(k.startswith(p) for p in skip_prefixes)}

    _ = model.load_state_dict(filtered, strict=True)
    _ = model.eval()
    model.is_train = False
    return model

def build_phmr_model(args, set_criterion=True):
    model_folder = "weights/pretrain/phmr"
    model = load_model_from_folder(model_folder)
    model.input_size = args.input_size
    if set_criterion:
        weight_dict = args.weight_dict
        losses = args.losses

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{k}.{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        criterion = SetCriterionHMR(weight_dict, losses = losses, j2ds_norm_scale = args.input_size)
        return model, criterion
    else:
        return model, None