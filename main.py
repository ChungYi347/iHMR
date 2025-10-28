import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_NO_DISTRIBUTED_DEBUG"] = "1"

import logging
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("torch.distributed").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)  #


import argparse
import yaml
import numpy as np
from engines.engine import Engine

def get_args_parser():
    parser = argparse.ArgumentParser('SAT-HMR', add_help=False)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--mode',default='train',type=str)
    parser.add_argument('--gpu_id',default=0,type=int)

    return parser

def update_args(args, cfg_path):
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
        args_dict = vars(args)
        args_dict.update(config)
        args = argparse.Namespace(**args_dict)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAT-HMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.cfg is not None
    args = update_args(args, os.path.join('configs', 'run', f'{args.cfg}.yaml'))
    args.exp_name = args.cfg

    if hasattr(args, 'model_type') and args.model_type == "sat_pr_sam2":
        args = update_args(args, os.path.join('configs', 'models', f'{args.model}_sam2.yaml'))
    else:
        args = update_args(args, os.path.join('configs', 'models', f'{args.model}.yaml'))
    
    


    if args.mode.lower() == 'train':
        from accelerate.utils import set_seed
        seed = args.seed
        set_seed(args.seed)
        engine = Engine(args, mode='train')
        engine.train()

    elif args.mode.lower() == 'eval':
        engine = Engine(args, mode='eval')
        engine.eval()

    elif args.mode.lower() == 'infer':
        engine = Engine(args, mode='infer')
        engine.infer()

    elif args.mode.lower() == 'eval_posetrack':
        engine = Engine(args, mode='eval_posetrack')
        engine.eval_posetrack(args)

    elif args.mode.lower() == 'eval_posetrack_video':
        engine = Engine(args, mode='eval_posetrack_video')
        engine.eval_posetrack_video(args)

    else:
        print('Wrong mode!')
        exit(1)