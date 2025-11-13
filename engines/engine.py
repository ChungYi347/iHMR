import accelerate
from packaging import version
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.multiple_datasets import MultipleDatasets, datasets_dict
from datasets.common import COMMON
from transformers import get_scheduler
from safetensors.torch import load_file
import os
import re
import time
import datetime
from models import build_sat_model, build_hmr_model, build_sat_pr_model, build_sat_pr_sam2_model_img, build_phmr_model, build_sat_video_model
from .funcs.eval_funcs import *
from .funcs.infer_funcs import inference, eval_posetrack, eval_posetrack_video, eval_3dpw_track, eval_bedlam_track
from utils import misc
from utils.misc import get_world_size
import torch.multiprocessing
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.process_batch import prepare_batch
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from datasets.pw3d import PW3D
from datasets.bedlam import BEDLAM

import json

import wandb

import shutil
from pathlib import Path
from collections import defaultdict

assert version.parse(accelerate.__version__) >= version.parse("1.10.0"),\
      "Please use accelerate >= 1.10.0 to support our updated implementation of distributed evaluation (August 30, 2025)."

class Engine():
    def __init__(self, args, mode='train'): 
        self.exp_name = args.exp_name
        self.mode = mode
        assert mode in ['train','eval','infer','eval_posetrack','eval_posetrack_video', 'eval_3dpw_track', 'eval_bedlam_track']
        self.conf_thresh = args.conf_thresh
        self.eval_func_maps = {'agora_validation': evaluate_agora,
                                'bedlam_validation_6fps': evaluate_agora,
                                'agora_test': test_agora,
                                '3dpw_test': evaluate_3dpw,
                                '3dpw_train': evaluate_3dpw,
                                '3dpw_video_test': evaluate_3dpw_video,}
        self.inference_func = inference
        self.wandb_id = None
        self.args = args

        if self.mode == 'train':
            self.output_dir = os.path.join('./outputs')
            self.log_dir = os.path.join(self.output_dir,'logs')
            self.ckpt_dir = os.path.join(self.output_dir,'ckpts')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'eval':
            self.output_dir = os.path.join('./results')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'infer':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer
        elif self.mode == 'eval_posetrack':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer
        elif self.mode == 'eval_posetrack_video':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer
        elif self.mode == 'eval_3dpw_track':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_eval
        elif self.mode == 'eval_bedlam_track':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_eval

        self.prepare_accelerator()
        self.prepare_models(args)
        self.prepare_datas(args)
        if self.mode == 'train':
            self.prepare_training(args)

        total_cnt = sum(p.numel() for p in self.model.parameters())
        trainable_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f'Initialization finished.\n{trainable_cnt} trainable parameters({total_cnt} total).')       

    def prepare_accelerator(self):
        if self.mode == 'train':
            self.accelerator = Accelerator(
                mixed_precision="bf16",   
                # gradient_accumulation_steps=getattr(self, "grad_accum_steps", 6),
                log_with="tensorboard",
                project_dir=os.path.join(self.log_dir)
            )
            if self.accelerator.is_main_process:
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(os.path.join(self.ckpt_dir,self.exp_name),exist_ok=True)
                self.accelerator.init_trackers(self.exp_name)
        else:
            self.accelerator = Accelerator()
            if self.accelerator.is_main_process:
                os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_models(self, args):
        # load model and criterion
        self.accelerator.print('Preparing models...')
        # self.unwrapped_model, self.criterion = build_phmr_model(args, set_criterion = (self.mode == 'train'))
        # self.unwrapped_model, self.criterion = build_sat_model(args, set_criterion = (self.mode == 'train'))
        # if args.model_type == "sat":
        #     self.unwrapped_model, self.criterion = build_sat_model(args, set_criterion = (self.mode == 'train'))
        # elif args.model_type == "sat_pr":
        #     self.unwrapped_model, self.criterion = build_sat_pr_model(args, set_criterion = (self.mode == 'train'))
        # elif args.model_type == "sat_pr_sam2":
        #     self.unwrapped_model, self.criterion = build_sat_pr_sam2_model(args, set_criterion = (self.mode == 'train'))
        # self.unwrapped_model, self.criterion = build_hmr_model(args, set_criterion = (self.mode == 'train'))
        # self.unwrapped_model, self.criterion = build_sat_pr_sam2_model(args, set_criterion = (self.mode == 'train'))
        if hasattr(args, 'video_model') and args.video_model:
            self.unwrapped_model, self.criterion = build_sat_video_model(args, set_criterion = (self.mode == 'train'))
        else:
            self.unwrapped_model, self.criterion = build_sat_pr_sam2_model_img(args, set_criterion = (self.mode == 'train'))
        
        if self.criterion is not None:
            self.weight_dict = self.criterion.weight_dict
        # load weights
        if args.pretrain:
            if args.encoder == "dinov3_vitb":
                self.accelerator.print(f'Loading pretrained weights: {args.pretrain_path}') 

                ckpt = torch.load(args.pretrain_path, map_location='cpu')
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

                model_dict = self.unwrapped_model.state_dict()

                filtered_state_dict = {}
                skipped = []

                for k, v in state_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            filtered_state_dict[k] = v
                        else:
                            skipped.append((k, v.shape, model_dict[k].shape))
                    else:
                        skipped.append((k, v.shape, None))

                model_dict.update(filtered_state_dict)
                self.unwrapped_model.load_state_dict(model_dict, strict=False)
            else:
                self.accelerator.print(f'Loading pretrained weights: {args.pretrain_path}') 
                state_dict = torch.load(args.pretrain_path, weights_only=False, map_location='cpu')
                self.unwrapped_model.load_state_dict(state_dict, strict=False)
            
        # to gpu
        self.model = self.accelerator.prepare(self.unwrapped_model)
        
    def prepare_datas(self, args):
        # load dataset and dataloader
        if self.mode == 'train':
            self.accelerator.print('Loading training datasets:\n',
                [f'{d}_{s}' for d,s in zip(args.train_datasets_used, args.train_datasets_split)])
            self.train_batch_size = args.train_batch_size

            args.aug = args.aug if hasattr(args, 'aug') else True
            if args.video_model:
                train_dataset = MultipleDatasets(args.train_datasets_used, args.train_datasets_split,
                                            make_same_len=False, input_size=args.input_size, aug=args.aug, 
                                            mode = 'train', sat_cfg=args.sat_cfg,
                                            aug_cfg=args.aug_cfg, backbone=args.encoder, sampling_weights=args.sampling_weights,
                                            video_stride=args.video_stride if hasattr(args, 'video_stride') else 1)
            else:
                train_dataset = MultipleDatasets(args.train_datasets_used, args.train_datasets_split,
                                            make_same_len=False, input_size=args.input_size, aug=args.aug, 
                                            mode = 'train', sat_cfg=args.sat_cfg,
                                            aug_cfg=args.aug_cfg, backbone=args.encoder, sampling_weights=args.sampling_weights)
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                                shuffle=True,collate_fn=misc.collate_fn, 
                                                num_workers=args.train_num_workers,pin_memory=True)
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)                                 

        if self.mode == 'eval_posetrack':
            pass
        elif self.mode == 'eval_posetrack_video':
            pass
        elif self.mode == 'eval_3dpw_track':
            pass
        elif self.mode == 'eval_bedlam_track':
            pass
        elif self.mode != 'infer':
            self.accelerator.print('Loading evaluation datasets:',
                                [f'{d}_{s}' for d,s in zip(args.eval_datasets_used, args.eval_datasets_split)])
            self.eval_batch_size = args.eval_batch_size
            if hasattr(args, 'video_model') and args.video_model:
                eval_ds = {f'{ds}_{split}': datasets_dict[ds](split = split, 
                                                            mode = 'eval', 
                                                            input_size = args.input_size, 
                                                            aug = False,
                                                            sat_cfg=args.sat_cfg,
                                                            backbone=args.encoder, 
                                                            video_stride=args.video_stride if hasattr(args, 'video_stride') else 1) 
                            for (ds, split) in zip(args.eval_datasets_used, args.eval_datasets_split)}
            else:
                eval_ds = {f'{ds}_{split}': datasets_dict[ds](split = split, 
                                                            mode = 'eval', 
                                                            input_size = args.input_size, 
                                                            aug = False,
                                                            sat_cfg=args.sat_cfg,
                                                            backbone=args.encoder) 
                            for (ds, split) in zip(args.eval_datasets_used, args.eval_datasets_split)}
            self.eval_dataloaders = {k: DataLoader(dataset=v, batch_size=self.eval_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.eval_num_workers,pin_memory=True)\
                                    for (k,v) in eval_ds.items()}
            if self.distributed_eval:
                for (k,v) in self.eval_dataloaders.items():
                    self.eval_dataloaders.update({k: self.accelerator.prepare(v)})
        else:
            img_folder = args.input_dir
            self.accelerator.print(f'Loading inference images from {img_folder}')
            self.infer_batch_size = args.infer_batch_size
            infer_ds = COMMON(img_folder = img_folder, input_size=args.input_size,aug=False,
                                mode = 'infer', sat_cfg=args.sat_cfg, backbone=args.encoder)
            self.infer_dataloader = DataLoader(dataset=infer_ds, batch_size=self.infer_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.infer_num_workers,pin_memory=True)

            if self.distributed_infer:
                self.infer_dataloader = self.accelerator.prepare(self.infer_dataloader)

    def prepare_training(self, args):
        self.start_epoch = 0
        self.num_epochs = args.num_epochs
        self.global_step = 0
        if hasattr(args, 'sat_gt_epoch'):
            self.sat_gt_epoch = args.sat_gt_epoch
            self.accelerator.print(f'Use GT for the first {self.sat_gt_epoch} epoch(s)...')
        else:
            self.sat_gt_epoch = -1
        self.save_and_eval_epoch = args.save_and_eval_epoch
        self.least_eval_epoch = args.least_eval_epoch

        self.detach_j3ds = args.detach_j3ds

        self.accelerator.print('Preparing optimizer and lr_scheduler...')   
        if hasattr(args, "lr_encoder_names") and (hasattr(args, "video_model") and not args.video_model):
            if hasattr(args, 'model_type') and args.model_type == "sat_pr":
                param_dicts = [
                    {
                        "params":
                            [p for n, p in self.unwrapped_model.named_parameters()
                            if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, ["prompt"])],
                        "lr": args.lr,
                    },
                    {
                        "params": 
                            [p for n, p in self.unwrapped_model.named_parameters() 
                            if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, ["prompt"])],
                        "lr": args.lr_encoder,
                    },
                    {
                        "params": 
                            [p for n, p in self.unwrapped_model.named_parameters() 
                            if misc.match_name_keywords(n, ["prompt"]) and p.requires_grad],
                        "lr": args.lr_propmt,
                    }
                ]
            # elif hasattr(args, 'model_type') and args.model_type == "sat_pr_sam2":
            #     param_dicts = [
            #         {
            #             "params":
            #                 [p for n, p in self.unwrapped_model.named_parameters()
            #                 if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, ["prompt"])],
            #             "lr": args.lr,
            #         },
            #         {
            #             "params": 
            #                 [p for n, p in self.unwrapped_model.named_parameters() 
            #                 if misc.match_name_keywords(n, ["prompt"]) and p.requires_grad],
            #             "lr": args.lr_propmt,
            #         }
            #     ]
            else:
                # param_dicts = [
                #     {
                #         "params":
                #             [p for n, p in self.unwrapped_model.named_parameters()
                #             if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                #         "lr": args.lr,
                #     },
                #     {
                #         "params": 
                #             [p for n, p in self.unwrapped_model.named_parameters() 
                #             if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                #         "lr": args.lr_encoder,
                #     },
                # ]
                # del (self.unwrapped_model.dn_enc)
                param_dicts = [
                    {
                        "params":
                            [p for n, p in self.unwrapped_model.named_parameters()
                            if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)],
                        "lr": args.lr,
                    },
                    {
                        "params": 
                            [p for n, p in self.unwrapped_model.named_parameters() 
                            if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)],
                        "lr": args.lr_encoder,
                    },
                    {
                        "params": 
                            [p for n, p in self.unwrapped_model.named_parameters() 
                            if misc.match_name_keywords(n, args.lr_prompt_names) and p.requires_grad],
                        "lr": args.lr_propmt,
                    },
                ]
                print([n for n, p in self.unwrapped_model.named_parameters() if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)])
                print([n for n, p in self.unwrapped_model.named_parameters() if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)])
                print([n for n, p in self.unwrapped_model.named_parameters() if misc.match_name_keywords(n, args.lr_prompt_names) and p.requires_grad])
        else:
            param_dicts = [
                {
                    "params":
                        [p for n, p in self.unwrapped_model.named_parameters()
                        if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)],
                    "lr": args.lr,
                },
                # {
                #     "params": 
                #         [p for n, p in self.unwrapped_model.named_parameters() 
                #         if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)],
                #     "lr": args.lr_encoder,
                # },
                {
                    "params": 
                        [p for n, p in self.unwrapped_model.named_parameters() 
                        if misc.match_name_keywords(n, args.lr_prompt_names) and p.requires_grad],
                    "lr": args.lr_propmt,
                },
            ]
            print([n for n, p in self.unwrapped_model.named_parameters() if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)])
            # print([n for n, p in self.unwrapped_model.named_parameters() if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad and not misc.match_name_keywords(n, args.lr_prompt_names)])
            print([n for n, p in self.unwrapped_model.named_parameters() if misc.match_name_keywords(n, args.lr_prompt_names) and p.requires_grad])

        # optimizer
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        
        # lr_scheduler
        if args.lr_scheduler == 'cosine':
            self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer, 
                                          num_warmup_steps=args.num_warmup_steps, 
                                          num_training_steps=get_world_size() * self.num_epochs * len(self.train_dataloader)) 
        elif args.lr_scheduler == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, gamma=args.gamma)  
        else:
            raise NotImplementedError      

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # resume
        if args.resume: #load model, optimizer, lr_scheduler and random_state
            if hasattr(args, 'ckpt_epoch'):
                self.load_ckpt(args.ckpt_epoch,args.ckpt_step)    
            else:
                self.accelerator.print('Auto resume from the latest ckpt...')
                epoch, step = -1, -1
                pattern = re.compile(r'epoch_(\d+)_step_(\d+)')
                for folder_name in os.listdir(os.path.join(self.output_dir,'ckpts',self.exp_name)):
                    match = pattern.match(folder_name)
                    if match:
                        i, j = int(match.group(1)), int(match.group(2))
                        if i > epoch:
                            epoch, step = i, j
                if epoch >= 0:
                    self.load_ckpt(epoch, step)    
                else:
                    self.accelerator.print('No existing ckpts! Train from scratch.')               

            if self.accelerator.is_main_process and hasattr(args, "wandb_logging") and args.wandb_logging:
                config = {k: v for k, v in args.weight_dict.items() if any(k == loss for loss in args.losses)}
                if self.wandb_id is None:
                    wandb.init(project="myproj", config=config, entity="chunggi_lee-harvard-university")
                else:
                    wandb.init(project="myproj", id=self.wandb_id, resume="must", config=config, entity="chunggi_lee-harvard-university")
                wandb.watch(self.model, log="parameters")
                self.wandb_id = None if wandb.run is None else wandb.run.id

    def load_ckpt(self, epoch, step):   
        self.accelerator.print(f'Loading checkpoint: epoch_{epoch}_step_{step}') 
        ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{step}')
        # import pdb; pdb.set_trace()
        # ckpts_save_path = os.path.join('/home/cglee/SAT-HMR/outputs/ckpts/train_all/epoch_13_step_31569')
        self.start_epoch = epoch + 1
        self.global_step = step + 1
        self.accelerator.load_state(ckpts_save_path)
        if os.path.exists(os.path.join(ckpts_save_path, 'config.pt')):
            loaded_config = torch.load(os.path.join(ckpts_save_path, 'config.pt'), map_location="cpu")
            self.wandb_id = loaded_config.wandb_id

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        # if self.model.__class__.__name__ == "PHMR" or self.model.module.__class__.__name__ == "PHMR":
        #     if not hasattr(self.model, 'module') and hasattr(self.model.prompt_encoder, 'clip_encoder'):
        #         del (self.model.prompt_encoder.clip_encoder)
        #     elif hasattr(self.model.module.prompt_encoder, 'clip_encoder'):
        #         del (self.model.module.prompt_encoder.clip_encoder)
        #     # prompt_encoder.mask_downscaling.1.bias
        #     # if hasattr('')

        # tracker = None
        # if self.model.__class__.__name__ == "VideoModel" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "VideoModel"):
        #     with torch.no_grad():
        #         import hydra
        #         from models.sam2.build_sam2 import build_sam2_video_predictor
        #         hydra.core.global_hydra.GlobalHydra.instance().clear()
        #         current_dir = os.path.dirname(os.path.abspath(__file__))
        #         config_dir = os.path.join(current_dir, "../sam2/configs/sam2.1")
        #         hydra.initialize_config_dir(config_dir=config_dir)
        #         checkpoint = "weights/sam2.1_hiera_large.pt"
        #         model_cfg = "sam2.1_hiera_l"
        #         tracker = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda', disable_compile=True)
        #         tracker.eval()

        # sam2_image_model = None
        # if self.model.__class__.__name__ == "ImageModel" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "ImageModel"):
        #     from models.sat_model_pr_sam2_img import SAM2ImageEncoder
        #     with torch.no_grad():
        #         sam2_model_cfg="sam2.1_hiera_l"
        #         sam2_checkpoint="weights/sam2.1_hiera_large.pt"
        #         enable_video = False

        #         sam2_image_model = SAM2ImageEncoder(sam2_model_cfg, sam2_checkpoint, enable_video)
        #         sam2_image_model.eval()


        self.accelerator.print('Start training!')
        for epoch in range(self.start_epoch, self.num_epochs):
            torch.cuda.empty_cache()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            self.model.train()
            self.criterion.train()

            sat_use_gt = (epoch < self.sat_gt_epoch)

            # for step, (samples,targets,masks) in enumerate(self.train_dataloader):
            for step, (samples,targets) in enumerate(self.train_dataloader):

                if self.model.__class__.__name__ == "PHMR" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "PHMR"):
                    inputs = []
                    for i, target in enumerate(targets):
                        img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
                        boxes = box_cxcywh_to_xyxy(target['boxes']).cpu()
                        if hasattr(self.model, 'module') and self.model.module.__class__.__name__  == "PHMR":
                            input_size = self.model.module.input_size
                        else:
                            input_size = self.model.input_size
                        boxes[:, 0] = boxes[:, 0] * input_size
                        boxes[:, 1] = boxes[:, 1] * input_size
                        boxes[:, 2] = boxes[:, 2] * input_size
                        boxes[:, 3] = boxes[:, 3] * input_size
                        # img = np.ascontiguousarray(img)
                        # for box in boxes.cpu().numpy():
                        #     box = box.astype(np.int32)
                        #     x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        #     cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
                        # cv2.imwrite('test2.jpg', img)
                        # import pdb; pdb.set_trace()
                        
                        
                        _input = {'image_cv': img, 'boxes': boxes, 'cam_int': target['cam_intrinsics'].cpu(), 'text': None, 'masks': None}
                        inputs.append(_input)
                    # inputs = [{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}]
                    
                    batch = prepare_batch(inputs, img_size=896, interaction=False)
                    # img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
                    outputs = self.model(batch, targets)
                elif self.model.__class__.__name__ == "VideoModel" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "VideoModel"):
                    loss_dicts = []
                    self.model.query_bank = []
                    if hasattr(self.model, 'module'):
                        self.model.module.query_bank = []
                    for img_seq_idx, (sample, target) in enumerate(zip(samples[0], targets[0])):
                        self.model.img_seq_idx = img_seq_idx
                        if hasattr(self.model, 'module'):
                            self.model.module.img_seq_idx = img_seq_idx
                        outputs = self.model(sample, target, sat_use_gt = sat_use_gt, detach_j3ds = self.detach_j3ds)
                        loss_dict = self.criterion(outputs, target)
                        loss_dicts.append(loss_dict)
                elif self.model.__class__.__name__ == "ImageModel" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "ImageModel"):
                    outputs = self.model(samples, targets, sat_use_gt = sat_use_gt, detach_j3ds = self.detach_j3ds)
                else:
                    outputs = self.model(samples, targets, sat_use_gt = sat_use_gt, detach_j3ds = self.detach_j3ds)
                # outputs = self.model(samples, targets, masks=masks, sat_use_gt = sat_use_gt, detach_j3ds = self.detach_j3ds)

                
                if self.model.__class__.__name__ == "VideoModel" or (hasattr(self.model, 'module') and self.model.module.__class__.__name__ == "VideoModel"):
                    loss_sums = defaultdict(float)
                    num_steps = len(loss_dicts)
                    for loss_dict in loss_dicts:
                        for k, v in loss_dict.items():
                            loss_sums[k] += v

                    loss_dicts = {k: v / num_steps for k, v in loss_sums.items()}
                else:
                    loss_dict = self.criterion(outputs, targets)

                loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())

                try:
                    self.accelerator.backward(loss)
                except:
                    print(loss)
                    raise

                # model = self.accelerator.unwrap_model(self.model) 
                # unused = [(i, n, tuple(p.shape))
                #         for i, (n, p) in enumerate(model.named_parameters())
                #         if p.requires_grad and (p.grad is None)]

                # self.accelerator.print(f"[step UNUSED params (no grad):")
                # for i, n, shp in unused[:50]:  
                #     self.accelerator.print(f"{i:4d}  {n:60s}  {shp}")
                # no_grad_params = []
                # for i, (n, p) in enumerate(model.named_parameters()):
                #     if p.requires_grad and p.grad is None:
                #         no_grad_params.append((i, n, tuple(p.shape)))
                # print(f"[RANK] params without grad:", no_grad_params[:20], " ...")



                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                reduced_dict = self.accelerator.reduce(loss_dict,reduction='mean')
                simplified_logs = {k: v.item() for k, v in reduced_dict.items() if '.' not in k}
                
                # logs.update({"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})
                if self.accelerator.is_main_process:
                    tqdm.write(f'[{epoch}-{step+1}/{len(self.train_dataloader)}]: ' + str(simplified_logs))
                    if hasattr(self.args, "wandb_logging") and self.args.wandb_logging:
                        wandb.log(simplified_logs)

                if step % 10 == 0:
                    self.accelerator.log({('train/'+k):v for k,v in simplified_logs.items()},
                                            step=self.global_step)
                    

                progress_bar.update(1)
                progress_bar.set_postfix(**{"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})

                self.global_step += 1
                self.accelerator.wait_for_everyone()

            self.lr_scheduler.step()

            if (epoch != 0 and epoch % self.save_and_eval_epoch == 0) or epoch == self.num_epochs-1:
                self.save_and_eval(epoch, save_ckpt=True)
            else:
                self.save_and_eval(epoch, save_ckpt=True, is_only_save=True)
        
        self.accelerator.end_training()

    def eval(self, results_save_path = None, epoch = -1):
        if results_save_path is None:
            results_save_path = os.path.join(self.output_dir,self.exp_name,'evaluation')
        # preparing
        self.model.eval()
        unwrapped_model = self.unwrapped_model # self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        # evaluate
        for i, (key, eval_dataloader) in enumerate(self.eval_dataloaders.items()):
            assert key in self.eval_func_maps
            img_cnt = len(eval_dataloader) * self.eval_batch_size
            if self.distributed_eval:
                img_cnt *= self.accelerator.num_processes
            self.accelerator.print(f'Evaluate on {key}: (about) {img_cnt} images')
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            conf_thresh = self.conf_thresh  # if 'agora' in key or 'bedlam' in key else [0.2]
            for thresh in conf_thresh:
                if self.accelerator.is_main_process or self.distributed_eval:
                    error_dict = self.eval_func_maps[key](model = unwrapped_model, 
                                    eval_dataloader = eval_dataloader, 
                                    conf_thresh = thresh,
                                    vis_step = img_cnt // self.eval_vis_num,
                                    results_save_path = os.path.join(results_save_path,key,f'thresh_{thresh}'),
                                    distributed = self.distributed_eval,
                                    accelerator = self.accelerator,
                                    vis=True)
                    if isinstance(error_dict,dict) and self.mode == 'train':
                        log_dict = flatten_dict(error_dict)
                        self.accelerator.log({(f'{key}_thresh_{thresh}/'+k):v for k,v in log_dict.items()}, step=epoch)

                    self.accelerator.print(f'thresh_{thresh}: ',error_dict)
                self.accelerator.wait_for_everyone() 

    def _parse_epoch(self, dir_name: str):
        EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)_step_(\d+)$")
        m = EPOCH_DIR_RE.match(dir_name)
        return int(m.group(1)) if m else None

    def _update_symlink(self, link_path: Path, target_path: Path):
        link_path = Path(link_path)
        target_path = Path(target_path).resolve()
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target_path, target_is_directory=True)
        except OSError:
            link_path.with_suffix(".txt").write_text(str(target_path))

    def _cleanup_keep_multiples_of_5(self, ckpts_root: Path, keep_epoch: int):
        if not ckpts_root.exists():
            return
        for d in ckpts_root.iterdir():
            if not d.is_dir(): 
                continue
            e = self._parse_epoch(d.name)
            if e is None:
                continue
            if (e % self.save_and_eval_epoch == 0) or (e == keep_epoch):
                continue
            try:
                shutil.rmtree(d)
            except Exception as ex:
                print(f"[ckpt cleanup] remove fail: {d} ({ex})")
  
    def save_and_eval(self, epoch, save_ckpt=False, is_only_save=False):
        torch.cuda.empty_cache()
        # save current state and model
        if self.accelerator.is_main_process and save_ckpt:
            ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')
            os.makedirs(ckpts_save_path,exist_ok=True)
            if self.wandb_id is not None:
                self.args.wandb_id = self.wandb_id
                torch.save(self.args, os.path.join(ckpts_save_path, 'config.pt'))
            self.accelerator.save_state(ckpts_save_path, safe_serialization=False)

            ckpts_root = Path(self.output_dir) / "ckpts" / self.exp_name
            self._update_symlink(ckpts_root / "last_epoch", ckpts_save_path)
            # if epoch % self.save_and_eval_epoch == 0:
            #     self._cleanup_keep_multiples_of_5(ckpts_root, keep_epoch=epoch)

        self.accelerator.wait_for_everyone()
        
        if not is_only_save:
            # if epoch < self.least_eval_epoch:
            #     return
            results_save_path = os.path.join(self.output_dir,'results',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')        
            self.eval(results_save_path, epoch=epoch)

    def infer(self):
        self.model.eval()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = self.unwrapped_model 
        
        results_save_path = self.output_dir
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        
        self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
        for thresh in self.conf_thresh:
            if self.accelerator.is_main_process or self.distributed_infer:
                self.inference_func(model = unwrapped_model, 
                        infer_dataloader = self.infer_dataloader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'thresh_{thresh}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator)
            self.accelerator.wait_for_everyone()

    def eval_posetrack(self, args):
        self.model.eval()
        self.model.is_tracking_eval = True

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        folder_list = sorted([
            os.path.join(args.input_dir, f) 
            for f in os.listdir(args.input_dir) 
            if os.path.isdir(os.path.join(args.input_dir, f))
        ])

        # target_folders = ['007934_mpii_test']
        # # target_folders = ['003742_mpii_test']
        # # target_folders = ['000809_mpii_test']
        # # # target_folders = ['004622_mpii_test', '015241_mpii_test']
        # # target_folders = ['009470_mpii_test']
        # 003742_mpii_test, 000814_mpii_test
        # target_folders = ['003508_mpii_test']
        # target_folders = ['005290_mpii_test']
        # target_folders = ['000812_mpii_test']
        # target_folders = ['024165_mpii_test', '015301_mpii_test', '016180_mpii_test',
        #                     '014140_mpii_test', '022691_mpii_test',
        #                     '002364_mpii_test']
        # target_folders = ['008827_mpii_test', '002835_mpii_test', '001486_mpii_test', '012834_mpii_test',
        #                     '016236_mpii_test', '007934_mpii_test', '024159_mpii_test', '024158_mpii_test',
        #                     '015860_mpii_test', '007934_mpii_test', '007793_mpii_test', '008827_mpii_test',
        #                     '012834_mpii_test', '011648_mpii_test']
        # target_folders = ['000707_mpii_test']
        
        # folder_list = [f for f in folder_list if any(target in f for target in target_folders)]
        # print(args.total_gpus)
        # folder_list = folder_list[96:]

        # assigned_folders = folder_list[args.gpu_id::args.total_gpus]
        assigned_folders = folder_list[args.gpu_id::1]
        if len(assigned_folders) == 0:
            self.accelerator.print(f"[Rank {rank}] No folders assigned. Skip.")
            return

        self.accelerator.print(f"[Rank {rank}] will process {len(assigned_folders)} folders")

        for i, img_folder in enumerate(assigned_folders):
            self.accelerator.print(f"[Rank {rank}] [{i+1}/{len(assigned_folders)}] {img_folder}")

            if hasattr(args, 'is_gt') and args.is_gt:
                tracking_gt_path = img_folder.replace('images', 'posetrack_data')+'.json'
                tracking_gt_path = tracking_gt_path.replace('Posetrack/', 'PoseTrack21/').replace('PoseTrack2018/', 'PoseTrack21/')

                anns = json.load(open(tracking_gt_path))
                images_ids = [i['image_id'] for i in  anns['images']]
                dic = {img_id:[[],[]] for img_id in images_ids}

                for ann in anns['annotations']:
                    dic[ann['image_id']][0].append(ann['track_id'])
                    dic[ann['image_id']][1].append(ann['bbox'])

                init_bboxes = []
                prev_ids = []
                prev_value = None
                for i, (k, v) in enumerate(dic.items()):
                    if v[0] == []:
                        continue
                    current_value = v[0]
                    not_in_b = list(set(current_value) - set(prev_ids))
                    prev_ids.extend(current_value)
                    prev_ids = list(set(prev_ids))

                    for idx in not_in_b:
                        init_bboxes.append([i, idx, v[1][v[0].index(idx)]])

            if hasattr(self, "tracker"):
                self.tracker.reset()
            seq_start = True

            infer_ds = COMMON(
                img_folder=img_folder,
                input_size=args.input_size,
                aug=False,
                mode='infer',
                sat_cfg=args.sat_cfg,
                backbone=args.encoder
            )

            infer_loader = DataLoader(
                dataset=infer_ds,
                batch_size=args.infer_batch_size,
                shuffle=False,
                collate_fn=misc.collate_fn,
                num_workers=args.infer_num_workers,
                pin_memory=True
            )

            # if self.distributed_infer:
            #     infer_loader = self.accelerator.prepare(infer_loader)

            unwrapped_model = self.unwrapped_model 

            from models.tracker import QueryTracker
            boxes_format = "cxcywh"  
            self.unwrapped_model.tracker = QueryTracker(self.args.conf_thresh[0], self.args.iou_match_thresh, 
                self.args.iou_new_thresh, self.args.max_age, boxes_format, is_add_new=self.args.is_add_new,
                is_size_filter=self.args.is_size_filter, nms_threshold=self.args.nms_threshold,
                pr_conf_thresh=self.args.pr_conf_thresh)
            
            results_save_path = self.output_dir
            if self.accelerator.is_main_process:
                os.makedirs(results_save_path,exist_ok=True)
            
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            img_folder = infer_loader.dataset.dataset_path.split('/')[-1]
            for thresh in self.conf_thresh:
                if not args.is_gt:
                    eval_posetrack(model = unwrapped_model, 
                            infer_dataloader = infer_loader, 
                            conf_thresh = thresh,
                            results_save_path = os.path.join(results_save_path,f'{img_folder}'),
                            distributed = self.distributed_infer,
                            accelerator = self.accelerator,
                            args = self.args)
                    # self.accelerator.wait_for_everyone()
                else:
                    eval_posetrack(model = unwrapped_model, 
                            infer_dataloader = infer_loader, 
                            conf_thresh = thresh,
                            results_save_path = os.path.join(results_save_path,f'{img_folder}'),
                            distributed = self.distributed_infer,
                            accelerator = self.accelerator,
                            args = self.args,
                            init_bboxes=init_bboxes)

    def eval_3dpw_track(self, args):
        self.model.eval()
        self.model.is_tracking_eval = True

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        folder_list = sorted([
            os.path.join(args.input_dir, f) 
            for f in os.listdir(args.input_dir) 
            if os.path.isdir(os.path.join(args.input_dir, f))
        ])
        print(folder_list)

        # target_folders = ['007934_mpii_test']
        # # target_folders = ['003742_mpii_test']
        # # target_folders = ['000809_mpii_test']
        # # # target_folders = ['004622_mpii_test', '015241_mpii_test']
        # # # target_folders = ['009470_mpii_test']
        target_folders = ['downtown_bar_00']
        folder_list = [f for f in folder_list if any(target in f for target in target_folders)]
        # print(args.total_gpus)

        assigned_folders = folder_list[args.gpu_id::args.total_gpus]
        # assigned_folders = folder_list[args.gpu_id::4]
        if len(assigned_folders) == 0:
            self.accelerator.print(f"[Rank {rank}] No folders assigned. Skip.")
            return

        self.accelerator.print(f"[Rank {rank}] will process {len(assigned_folders)} folders")

        for i, img_folder in enumerate(assigned_folders):
            self.accelerator.print(f"[Rank {rank}] [{i+1}/{len(assigned_folders)}] {img_folder}")

            if hasattr(self, "tracker"):
                self.tracker.reset()
            seq_start = True

            # infer_ds = PW3D(
            #     img_folder=img_folder,
            #     input_size=args.input_size,
            #     aug=False,
            #     mode='infer',
            #     sat_cfg=args.sat_cfg,
            #     backbone=args.encoder
            # )

            # infer_loader = DataLoader(
            #     dataset=infer_ds,
            #     batch_size=args.infer_batch_size,
            #     shuffle=False,
            #     collate_fn=misc.collate_fn,
            #     num_workers=args.infer_num_workers,
            #     pin_memory=True
            # )

            infer_ds = PW3D(split = 'test', 
                            mode = 'eval', 
                            input_size = args.input_size, 
                            aug = False,
                            sat_cfg=args.sat_cfg,
                            backbone=args.encoder,
                            target_folders=img_folder) 
            infer_loader = DataLoader(dataset=infer_ds, batch_size=args.eval_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.eval_num_workers,pin_memory=True)
           
            # if self.distributed_infer:
            #     infer_loader = self.accelerator.prepare(infer_loader)

            unwrapped_model = self.unwrapped_model 

            from models.tracker import QueryTracker
            boxes_format = "cxcywh"  
            self.unwrapped_model.tracker = QueryTracker(self.args.conf_thresh[0], self.args.iou_match_thresh, self.args.iou_new_thresh, 
                self.args.max_age, boxes_format, is_add_new=self.args.is_add_new,
                is_size_filter=self.args.is_size_filter, nms_threshold=self.args.nms_threshold,
                pr_conf_thresh=self.args.pr_conf_thresh)
            
            results_save_path = self.output_dir
            if self.accelerator.is_main_process:
                os.makedirs(results_save_path,exist_ok=True)
            
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            img_folder = infer_loader.dataset.target_folders.split('/')[-1]
            for thresh in self.conf_thresh:
                eval_3dpw_track(model = unwrapped_model, 
                        infer_dataloader = infer_loader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'{img_folder}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator,
                        args = self.args)
                # self.accelerator.wait_for_everyone()

    def eval_bedlam_track(self, args):
        self.model.eval()
        self.model.is_tracking_eval = True

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        folder_list = sorted([
            os.path.join(args.input_dir, f) 
            for f in os.listdir(args.input_dir) 
            if os.path.isdir(os.path.join(args.input_dir, f))
        ])
        print(folder_list)

        # target_folders = ['007934_mpii_test']
        # # target_folders = ['003742_mpii_test']
        # # target_folders = ['000809_mpii_test']
        # # # target_folders = ['004622_mpii_test', '015241_mpii_test']
        # # # target_folders = ['009470_mpii_test']
        target_folders = ['20221019_3-8_250_highbmihand_orbit_stadium_6fps']
        # target_folders = ['20221019_3-8_250_highbmihand_orbit_stadium_6fps/png/seq_000004']
        folder_list = [f for f in folder_list if any(target in f for target in target_folders)]
        # print(args.total_gpus)

        assigned_folders = folder_list[args.gpu_id::1]
        # assigned_folders = folder_list[args.gpu_id::4]
        if len(assigned_folders) == 0:
            self.accelerator.print(f"[Rank {rank}] No folders assigned. Skip.")
            return

        self.accelerator.print(f"[Rank {rank}] will process {len(assigned_folders)} folders")

        for i, img_folder in enumerate(assigned_folders):
            self.accelerator.print(f"[Rank {rank}] [{i+1}/{len(assigned_folders)}] {img_folder}")

            if hasattr(self, "tracker"):
                self.tracker.reset()
            seq_start = True

            # infer_ds = PW3D(
            #     img_folder=img_folder,
            #     input_size=args.input_size,
            #     aug=False,
            #     mode='infer',
            #     sat_cfg=args.sat_cfg,
            #     backbone=args.encoder
            # )

            # infer_loader = DataLoader(
            #     dataset=infer_ds,
            #     batch_size=args.infer_batch_size,
            #     shuffle=False,
            #     collate_fn=misc.collate_fn,
            #     num_workers=args.infer_num_workers,
            #     pin_memory=True
            # )

            infer_ds = BEDLAM(split = 'validation_6fps', 
                            mode = 'eval', 
                            input_size = args.input_size, 
                            aug = False,
                            sat_cfg=args.sat_cfg,
                            backbone=args.encoder,
                            target_folders=img_folder) 
            infer_loader = DataLoader(dataset=infer_ds, batch_size=args.eval_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.eval_num_workers,pin_memory=True)
           
            # if self.distributed_infer:
            #     infer_loader = self.accelerator.prepare(infer_loader)

            unwrapped_model = self.unwrapped_model 

            from models.tracker import QueryTracker
            boxes_format = "cxcywh"  
            self.unwrapped_model.tracker = QueryTracker(self.args.conf_thresh[0], self.args.iou_match_thresh, self.args.iou_new_thresh, 
                self.args.max_age, boxes_format, is_add_new=self.args.is_add_new,
                is_size_filter=self.args.is_size_filter, nms_threshold=self.args.nms_threshold,
                pr_conf_thresh=self.args.pr_conf_thresh)
            
            results_save_path = self.output_dir
            if self.accelerator.is_main_process:
                os.makedirs(results_save_path,exist_ok=True)
            
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            img_folder = infer_loader.dataset.target_folders.split('/')[-1]
            for thresh in self.conf_thresh:
                eval_bedlam_track(model = unwrapped_model, 
                        infer_dataloader = infer_loader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'{img_folder}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator,
                        args = self.args)
                # self.accelerator.wait_for_everyone()

    def eval_posetrack_video(self, args):
        self.model.eval()
        self.model.is_tracking_eval = True

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        folder_list = sorted([
            os.path.join(args.input_dir, f) 
            for f in os.listdir(args.input_dir) 
            if os.path.isdir(os.path.join(args.input_dir, f))
        ])

        # target_folders = ['016236_mpii_test', '018092_mpii_test', '002214_mpii_test', '002371_mpii_test']
        # target_folders = ['004622_mpii_test', '015241_mpii_test']
        # target_folders = ['024165_mpii_test']
        # import pdb; pdb.set_trace()
        # folder_list = [f for f in folder_list if any(target in f for target in target_folders)]
        # # print(args.total_gpus)

        assigned_folders = folder_list[args.gpu_id::args.total_gpus]
        if len(assigned_folders) == 0:
            self.accelerator.print(f"[Rank {rank}] No folders assigned. Skip.")
            return

        # sam2_tracker = None
        # with torch.no_grad():
        #     import hydra
        #     from models.sam2.build_sam2 import build_sam2_video_predictor
        #     hydra.core.global_hydra.GlobalHydra.instance().clear()
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     config_dir = os.path.join(current_dir, "../sam2/configs/sam2.1")
        #     hydra.initialize_config_dir(config_dir=config_dir)
        #     checkpoint = "weights/sam2.1_hiera_large.pt"
        #     model_cfg = "sam2.1_hiera_l"
        #     sam2_tracker = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda', disable_compile=True)
        #     sam2_tracker.eval()

        self.accelerator.print(f"[Rank {rank}] will process {len(assigned_folders)} folders")

        for i, img_folder in enumerate(assigned_folders):
            self.accelerator.print(f"[Rank {rank}] [{i+1}/{len(assigned_folders)}] {img_folder}")

            if hasattr(self, "tracker"):
                self.tracker.reset()
            seq_start = True

            infer_ds = COMMON(
                img_folder=img_folder,
                input_size=args.input_size,
                aug=False,
                mode='infer',
                sat_cfg=args.sat_cfg,
                backbone=args.encoder
            )

            infer_loader = DataLoader(
                dataset=infer_ds,
                batch_size=args.infer_batch_size,
                shuffle=False,
                collate_fn=misc.collate_fn,
                num_workers=args.infer_num_workers,
                pin_memory=True
            )

            # if self.distributed_infer:
            #     infer_loader = self.accelerator.prepare(infer_loader)

            unwrapped_model = self.unwrapped_model 

            from models.tracker import QueryTracker
            boxes_format = "cxcywh"  
            self.unwrapped_model.tracker = QueryTracker(self.args.conf_thresh[0], self.args.iou_match_thresh, 
                self.args.iou_new_thresh, self.args.max_age, boxes_format, is_add_new=self.args.is_add_new,
                is_size_filter=self.args.is_size_filter, nms_threshold=self.args.nms_threshold,
                pr_conf_thresh=self.args.pr_conf_thresh)
            
            results_save_path = self.output_dir
            if self.accelerator.is_main_process:
                os.makedirs(results_save_path,exist_ok=True)
            
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            img_folder = infer_loader.dataset.dataset_path.split('/')[-1]
            for thresh in self.conf_thresh:
                eval_posetrack_video(model = unwrapped_model, 
                        infer_dataloader = infer_loader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'{img_folder}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator,
                        args = self.args)
                # self.accelerator.wait_for_everyone()


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

