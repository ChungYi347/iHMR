# Modified from DINOv2 (https://github.com/facebookresearch/dinov2)
from models.encoders.dinov2.models.vision_transformer import vit_base, vit_large
from models.encoders.dinov2.models.dinov3 import dinov3_vit_base
import torch
from configs.paths import dinov2_vitb14_path, dinov2_vitl14_path, dinov3_vitb16_path
import copy

REPO_DIR = "/home/cglee/dinov3"

def build_encoder(args):
    num_additional_blocks = 0
    if args.sat_cfg['use_sat'] and args.sat_cfg['use_additional_blocks']:
        num_additional_blocks = args.sat_cfg['get_map_layer']

    weights = None
    if args.encoder == 'vitb':
        model = vit_base(img_size = 518,
            patch_size  = 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens = 0,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            num_additional_blocks = num_additional_blocks)
        if args.mode.lower() == 'train':
            weights = torch.load(dinov2_vitb14_path, weights_only=False)
    elif args.encoder == 'vitl':
        model = vit_large(img_size = 518,
            patch_size  = 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens = 0,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            num_additional_blocks = num_additional_blocks)
        if args.mode.lower() == 'train':
            weights = torch.load(dinov2_vitl14_path, weights_only=False)
    elif args.encoder == "dinov3_vitb":
        print('Loading pretrained DINOv3...')
        model = dinov3_vit_base(img_size = 518,
            patch_size  = 16,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens = 0,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            num_additional_blocks = num_additional_blocks)
        if args.mode.lower() == 'train':
            weights_dinov2 = torch.load(dinov2_vitb14_path, weights_only=False)
            weights = torch.load(dinov3_vitb16_path, weights_only=False)
            from torch import nn
            weights['pos_embed'] = nn.Parameter(torch.zeros(1, 1025, 768))
            model.storage_tokens.requires_grad=False
        # model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=dinov3_vitb16_path)
    else:
        raise NotImplementedError
    if weights is not None:
        if args.sat_cfg['use_sat'] and args.sat_cfg['use_additional_blocks']:
            add_blocks_weights(weights, args.sat_cfg['get_map_layer'])
        print('Loading pretrained DINOv2...')
        model.load_state_dict(weights,strict=True)

    return model

def add_blocks_weights(weights, num_layers):
    for k in list(weights.keys()):
        if k.startswith('blocks') and int(k.split('.')[1]) < num_layers:
            new_k = k.replace('blocks', 'additional_blocks')
            weights[new_k] = copy.deepcopy(weights[k])
