# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import os

import cv2
import math
from math import tan,pi
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize
import numpy as np
import time
import random
import hydra

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from utils.transforms import rot6d_to_axis_angle, img2patch_flat, img2patch, to_zorder
from utils.map import build_z_map
from utils import constants
from configs.paths import smpl_mean_path

from models.encoders import build_encoder
from .matcher import build_matcher
from .decoder import build_decoder
from .position_encoding import position_encoding_xy
from .criterion import SetCriterionSAM2
from .dn_components import prepare_for_cdn, dn_post_process
import copy

from configs.paths import smpl_model_path


import os, sys
sys.path.append(os.path.abspath("./sam2"))

import copy
import pdb
import os
import math
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from .utils import MLP, rotation_6d_to_matrix, matrix_to_axis_angle

from models.sam2.build_sam2 import build_sam2
from models.sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import mask_to_box
from .position_encoding import build_position_encoding

from PIL import Image

from .prompt import SMPLDecoder, SMPLXDecoder, CameraEncoder
# from .prompt import PromptEncoder as PHMRPromptEncoder  # Use SAM2's original prompt encoder instead
from .prompt.common import LayerNorm2d

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from models.human_models import SMPL_Layer, smpl_gendered
from utils.transforms import rot6d_to_axis_angle


class SAM2ImageEncoder(nn.Module):
    """SAM2-based image encoder"""
    def __init__(self, model_cfg="sam2.1_hiera_l", checkpoint="weights/sam2.1_hiera_large.pt", enable_video=False):
        super().__init__()
        self.enable_video = enable_video
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, "../sam2/configs/sam2.1")
        hydra.initialize_config_dir(config_dir=config_dir)
        
        if not self.enable_video:
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
                self.sam2_predictor = SAM2ImagePredictor(sam_model=build_sam2(model_cfg, checkpoint, device=f'cuda:{local_rank}'))
            else:
                self.sam2_predictor = SAM2ImagePredictor(sam_model=build_sam2(model_cfg, checkpoint))

        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    
    def forward(self, imgs, body_bboxes, img_shapes, device, targets=None, detector=None, input_size=1288):
        target = targets[0] if targets and len(targets) > 0 else None

        if device != self.mean.device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device) 
        
        det_body_bboxes = []
        imgs_list = []
        input_boxes = []
        for img_idx in range(len(imgs)):
            img = imgs[img_idx] * self.std.to(device) + self.mean.to(device)
            img = torch.clamp(img, 0, 255).to(torch.uint8)
            
            img_shape = img_shapes[img_idx]
            img = img.permute(1, 2, 0)  # (1, H, W, 3)
            img_np = img.cpu().numpy()
            img_h, img_w = img_shape[0], img_shape[1]
            imgs_list.append(img_np)

            if detector != None:
                body_bbox = torch.tensor(detector(img_np)[0]).to(img.device)
                det_body_bboxes.append(body_bbox)
        
        self.sam2_predictor.reset_predictor()
        self.sam2_predictor.set_image_batch(imgs_list)
        
        # if len(body_bbox) == 0:
        #     input_boxes = torch.tensor([[0, 0, 0, 0]])
        if detector != None:
            for body_bbox in det_body_bboxes:
                input_box = torch.cat([
                    body_bbox[:, [0]],  
                    body_bbox[:, [1]],  
                    body_bbox[:, [2]],  
                    body_bbox[:, [3]]   
                ], dim=1)
                input_boxes.append(input_box)
        else:
            # Convert normalized coordinates to pixel coordinates
            for img_id in range(len(body_bboxes)):
                # # img_shape = img_shapes[img_idx]
                # # img_h, img_w = img_shape[0], img_shape[1]
                # body_bbox = box_cxcywh_to_xyxy(body_bboxes[img_id]) * input_size
                # input_box= torch.cat([
                #     body_bbox[:, [0]],  
                #     body_bbox[:, [1]],  
                #     body_bbox[:, [2]],  
                #     body_bbox[:, [3]]  
                # ], dim=1)
                # input_boxes.append(input_box)
                img_shape = img_shapes[img_idx]
                img_h, img_w = img_shape[0], img_shape[1]
                body_bbox = box_cxcywh_to_xyxy(body_bboxes[img_id]) * input_size
                input_box= torch.cat([
                    body_bbox[:, [0]],  
                    body_bbox[:, [1]],  
                    body_bbox[:, [2]],  
                    body_bbox[:, [3]]
                ], dim=1)
                input_boxes.append(input_box)
            
            # # Add Gaussian noise and box transformations during training for better generalization
            # if self.training:
            #     # Randomly choose box type: whole-body or truncated
            #     box_type = torch.randint(0, 2, (1,)).item()  # 0: whole-body, 1: truncated
                
            #     if box_type == 0:  # Truncated box - random crop
            #         # Randomly crop 70-90% of the original box
            #         crop_ratio = 0.9 + torch.rand(1).item() * 0.1  # 0.9 to 1.0
            #         width = input_boxes[:, 2] - input_boxes[:, 0]
            #         height = input_boxes[:, 3] - input_boxes[:, 1]
                    
            #         new_width = width * crop_ratio
            #         new_height = height * crop_ratio
                    
            #         # Random offset within original box
            #         offset_x = torch.rand(1).item() * (width - new_width)
            #         offset_y = torch.rand(1).item() * (height - new_height)
                    
            #         input_boxes[:, 0] = input_boxes[:, 0] + offset_x
            #         input_boxes[:, 1] = input_boxes[:, 1] + offset_y
            #         input_boxes[:, 2] = input_boxes[:, 0] + new_width
            #         input_boxes[:, 3] = input_boxes[:, 1] + new_height
            #     else:
            #         # Add Gaussian noise to both corners (5% of bbox size)
            #         bbox_width = input_boxes[:, 2] - input_boxes[:, 0]
            #         bbox_height = input_boxes[:, 3] - input_boxes[:, 1]
            #         noise_std = 0.05  # 5% of bbox size
                    
            #         noise_x = torch.randn_like(input_boxes[:, [0, 2]]) * noise_std * bbox_width.unsqueeze(-1)
            #         noise_y = torch.randn_like(input_boxes[:, [1, 3]]) * noise_std * bbox_height.unsqueeze(-1)
                    
            #         input_boxes[:, [0, 2]] = input_boxes[:, [0, 2]] + noise_x
            #         input_boxes[:, [1, 3]] = input_boxes[:, [1, 3]] + noise_y
                    
            #         # Clamp to valid image bounds
            #         input_boxes[:, [0, 2]] = torch.clamp(input_boxes[:, [0, 2]], 0, img_w)
            #         input_boxes[:, [1, 3]] = torch.clamp(input_boxes[:, [1, 3]], 0, img_h)
                    
            #         # Ensure x1 < x2 and y1 < y2
            #         input_boxes[:, [0, 2]] = torch.sort(input_boxes[:, [0, 2]], dim=1)[0]
            #         input_boxes[:, [1, 3]] = torch.sort(input_boxes[:, [1, 3]], dim=1)[0]
        self.input_boxes = input_boxes
        
        if self.training:
            # Randomly choose input type: 0 for box, 1 for mask
            input_type = torch.randint(0, 2, (1,)).item()
        else:
            # During inference, always use box
            input_type = 0

        # import matplotlib.pyplot as plt
        # def show_box(box, ax):
        #     x0, y0 = box[0], box[1]
        #     w, h = box[2] - box[0], box[3] - box[1]
        #     print(x0, y0, w, h)
        #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

        # def show_mask(mask, ax, obj_id=None, random_color=False):
        #     if random_color:
        #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        #     else:
        #         cmap = plt.get_cmap("tab10")
        #         cmap_idx = 0 if obj_id is None else obj_id
        #         color = np.array([*cmap(cmap_idx)[:3], 0.6])
        #     h, w = mask.shape[-2:]
        #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        #     ax.imshow(mask_image)

        # masks, scores, _ = self.sam2_predictor.predict(
        masks, scores, _, all_sparse_embeddings, all_dense_embeddings = self.sam2_predictor.predict_batch(
            point_coords_batch=None,
            point_labels_batch=None,
            box_batch=input_boxes,
            multimask_output=False,
        )

        # import matplotlib.pyplot as plt
        # for img_id in range(len(body_bboxes)):
        #     plt.figure(figsize=(9, 9))
        #     plt.imshow(imgs_list[img_id])
        #     for n, _bbox in enumerate(input_boxes[img_id].cpu().numpy()):
        #         show_mask((masks[img_id][n] > 0.0), plt.gca(), obj_id=n)
        #         show_box(_bbox, plt.gca())
        #     print(f'test_mask_all_{img_id}_{n}.png')
        #     plt.savefig(f'test_mask_all_{img_id}_{n}.png')
        #     plt.close()
        # import pdb; pdb.set_trace()
        
        # if input_type == 0:
        #     # Use bounding box as input
        #     masks, scores, _ = self.sam2_predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         box=input_boxes,
        #         multimask_output=False,
        #     )
        # else:
        #     # Use mask as input - 2-stage SAM2 prediction
        #     # Stage 1: Generate mask using bbox
        #     stage1_masks, stage1_scores, stage1_logits = self.sam2_predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         box=input_boxes,
        #         multimask_output=False,
        #     )
            
        #     # Use the first (and only) mask logits
        #     if len(stage1_masks) > 0:
        #         mask_input = stage1_logits[0, :, :]  # Use the first mask logits
        #     else:
        #         # Fallback: create simple mask from bbox if stage 1 failed
        #         mask_input = np.zeros((img_h, img_w), dtype=np.uint8)
        #         if len(input_boxes) > 0 and input_boxes[0, 2] > input_boxes[0, 0] and input_boxes[0, 3] > input_boxes[0, 1]:
        #             x1, y1, x2, y2 = input_boxes[0].int().cpu().numpy()
        #             mask_input[y1:y2, x1:x2] = 1
            
        #     # Stage 2: Refine using mask input (logits)
        #     masks, scores, _ = self.sam2_predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         mask_input=mask_input[None, :, :],  # Add batch dimension
        #         multimask_output=False,
        #     )
        
        sam2_feats = [
            self.sam2_predictor._features['features'][0],  # 256x256
            self.sam2_predictor._features['features'][1],  # 128x128  
            self.sam2_predictor._features['features'][2],  # 64x64
            self.sam2_predictor._features['features'][3],  # 32x32
        ]
        
        return sam2_feats, all_sparse_embeddings, all_dense_embeddings # , sam2_boxes, sam2_masks
    
    def convert_boxes(self, sam_boxes, img_size):
        """Convert bounding boxes"""
        h, w = img_size
        sam_boxes = sam_boxes.to(torch.float32).to(img_size.device)
        x_min, y_min, x_max, y_max = sam_boxes.unbind(-1)

        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        return torch.stack([x_center, y_center, width, height], dim=-1)


# class SAM2PromptEncoder(nn.Module):
#     """SAM2-based prompt encoder - integrates PHMR PromptEncoder with SAM2 features"""
#     def __init__(self, embed_dim=1024, image_embedding_size=(64, 64), input_image_size=(1024, 1024),
#                  text_prompt=True, kpt_prompt=True, mask_prompt=True, box_prompt=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.image_embedding_size = image_embedding_size
#         self.input_image_size = input_image_size
#         
#         # Initialize PHMR PromptEncoder
#         self.phmr_prompt_encoder = PHMRPromptEncoder(
#             embed_dim=embed_dim,
#             image_embedding_size=image_embedding_size,
#             input_image_size=input_image_size,
#             clip_encoder=None,  # Add if needed
#             text_prompt=text_prompt,
#             kpt_prompt=kpt_prompt,
#             mask_prompt=mask_prompt
#         )
#         
#         # Adapter to convert SAM2 features to PHMR format
#         self.sam2_adapter = nn.Sequential(
#             nn.Conv2d(256, embed_dim, 1),
#             nn.LayerNorm([embed_dim, image_embedding_size[0], image_embedding_size[1]]),
#             nn.GELU()
#         )
#     
#     def forward(self, sam2_features, boxes_list=None, kpts_list=None, text_list=None, masks_list=None):
#         """SAM2 features to prompt encoding for image model (batch processing)"""
#         # Use SAM2's 3rd resolution (index 2) which is 64x64, same as DINOv2
#         # sam2_features[2] has shape [B, 256, 64, 64] - no need to resize
#         target_size = (64, 64)
#         resized_features = []
#         
#         for feat in sam2_features:
#             feat_resized = F.interpolate(feat[0], size=target_size, mode='bilinear', align_corners=False)
#             resized_features.append(feat_resized)
#         
#         # 4개를 concat하여 1024 채널 만들기
#         # [B, 256, 32, 32] + [B, 256, 32, 32] + [B, 256, 32, 32] + [B, 256, 32, 32]
#         # = [B, 1024, 32, 32]
#         image_embeddings = torch.cat(resized_features, dim=1)  # [B, 1024, 32, 32]
#         
#         # Process prompts for batch (handle None values)
#         if boxes_list is not None:
#             # Filter out None values and process valid boxes
#             valid_boxes = [boxes for boxes in boxes_list if boxes is not None]
#             if valid_boxes:
#                 boxes = torch.cat(valid_boxes, dim=0)
#                 # Add confidence score (1.0) to boxes for PHMR compatibility
#                 # PHMR expects [x1, y1, x2, y2, conf] format
#                 conf = torch.ones(boxes.shape[0], 1, device=boxes.device)
#                 boxes = torch.cat([boxes, conf], dim=-1)
#             else:
#                 boxes = None
#         else:
#             boxes = None
#             
#         if kpts_list is not None:
#             valid_kpts = [kpts for kpts in kpts_list if kpts is not None]
#             if valid_kpts:
#                 kpts = torch.cat(valid_kpts, dim=0)
#             else:
#                 kpts = None
#         else:
#             kpts = None
#             
#         if text_list is not None:
#             valid_text = [text for text in text_list if text is not None]
#             if valid_text:
#                 text = valid_text[0]  # Use first valid text for now
#             else:
#                 text = None
#         else:
#             text = None
#             
#         if masks_list is not None:
#             valid_masks = [masks for masks in masks_list if masks is not None]
#             if valid_masks:
#                 masks = torch.cat(valid_masks, dim=0)
#             else:
#                 masks = None
#         else:
#             masks = None
#         
#         # Use PHMR PromptEncoder
#         sparse_embeddings, dense_embeddings = self.phmr_prompt_encoder(
#             boxes, text, kpts, masks
#         )
#         
#         return image_embeddings, sparse_embeddings, dense_embeddings
#     
#     def get_dense_pe(self):
#         """Return dense positional encoding"""
#         return self.phmr_prompt_encoder.get_dense_pe()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C



class SAM2PromptEncoder(nn.Module):
    """SAM2-based prompt encoder using SAM2ImagePredictor's built-in prompt encoder"""
    def __init__(self, embed_dim=1024, image_embedding_size=(64, 64), input_image_size=(1024, 1024),
                 text_prompt=True, kpt_prompt=True, mask_prompt=True, box_prompt=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        
        # We'll use the prompt encoder from SAM2ImagePredictor directly
        # No need to create a separate one
        
        # Adapter to convert SAM2 features to target format (1024 -> 256)
        self.sam2_adapter = nn.Sequential(
            # nn.Conv2d(embed_dim, embed_dim//4, 1, bias=False),
            # LayerNorm2d(embed_dim//4),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            # LayerNorm2d(embed_dim),
            # nn.GroupNorm(32, embed_dim),  
            # nn.SiLU(),
        )

        self.target_size = (64, 64)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
    
    def forward(self, sam2_features, boxes_list=None, kpts_list=None, text_list=None, masks_list=None):
        """Use SAM2ImagePredictor's built-in prompt encoder"""
        # Process SAM2 features - resize and concatenate
        resized_features = []
        
        for feat in sam2_features:
            feat_resized = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat_resized)
        
        # Concatenate features to create image embeddings
        image_embeddings = torch.cat(resized_features, dim=1)  # [B, 1024, 64, 64]
        
        # Apply adapter to convert 1024 -> 256 channels
        image_embeddings = self.sam2_adapter(image_embeddings)  # [B, 256, 64, 64]
        
        # # Process prompts for batch (handle None values)
        # if boxes_list is not None:
        #     # Filter out None values and process valid boxes
        #     valid_boxes = [boxes for boxes in boxes_list if boxes is not None]
        #     if valid_boxes:
        #         boxes = torch.cat(valid_boxes, dim=0)
        #         # Convert to SAM2 format: [x1, y1, x2, y2] (no confidence needed for SAM2)
        #         boxes = boxes[:, :4]  # Remove confidence if present
        #     else:
        #         boxes = None
        # else:
        #     boxes = None
            
        # # For now, we'll only use boxes as SAM2's original prompt encoder supports points, boxes, and masks
        # # Text and keypoints are not directly supported by SAM2's original prompt encoder
        # points = None
        # masks = None
        
        # if masks_list is not None:
        #     valid_masks = [masks for masks in masks_list if masks is not None]
        #     if valid_masks:
        #         masks = torch.cat(valid_masks, dim=0)
        #     else:
        #         masks = None
        # else:
        #     masks = None
        
        # # Use SAM2ImagePredictor's built-in prompt encoder
        # sparse_embeddings, dense_embeddings = sam2_image_encoder.sam2_predictor.model.sam_prompt_encoder(
        #     points=points,
        #     boxes=boxes,
        #     masks=masks
        # )
        
        # return image_embeddings, sam2_image_encoder.sam2_predictor.sparse_embeddings, sam2_image_encoder.sam2_predictor.dense_embeddings
        return image_embeddings
    
    def get_dense_pe(self):
        """Return dense positional encoding from SAM2ImagePredictor"""
        return self.pe_layer(self.target_size).unsqueeze(0)


class SAM2SMPLDecoder(nn.Module):
    """SAM2 feature-based SMPL decoder - based on PHMR SMPLXDecoder"""
    def __init__(self, transformer_dim=256, smpl_head_depth=2, smpl_head_hidden_dim=256, is_init_pose=False):
        super().__init__()
        self.transformer_dim = transformer_dim
        
        # Initialize PHMR SMPLXDecoder (SMPLX with hands, face, expression)
        self.smplx_decoder = SMPLXDecoder(
            transformer_dim=transformer_dim,
            smplx_head_depth=smpl_head_depth,
            smplx_head_hidden_dim=smpl_head_hidden_dim,
            use_hands=True,
            use_face=True,
            use_expression=False,
            is_init_pose=is_init_pose
        )
        
        # Keep original SMPL decoder as backup (commented out)
        # self.smpl_decoder = SMPLDecoder(
        #     transformer_dim=transformer_dim,
        #     transformer=self.transformer,
        #     smpl_head_depth=smpl_head_depth,
        #     smpl_head_hidden_dim=smpl_head_hidden_dim,
        #     inverse_depth=True
        # )
    
    def forward(self, cam_int, image_embeddings, image_pe, sparse_prompt_embeddings, 
                dense_prompt_embeddings, crossperson=False):
        """Predict SMPLX parameters"""
        return self.smplx_decoder(
            cam_int, image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, crossperson
        )


class Sam2Model(nn.Module):
    """PHMR-style SAM2AiOSSMPLX model - integrates SAM2 + PHMR components"""
    
    def __init__(
        self,
        input_size,
        # Basic settings
        hidden_dim=256,
        
        # SAM2 settings
        sam2_model_cfg="sam2.1_hiera_l",
        sam2_checkpoint="weights/sam2.1_hiera_large.pt",
        enable_video=False,
        
        # PHMR settings
        img_size=1024,
        embed_dim=1024,
        image_embedding_size=(64, 64),
        text_prompt=True,
        kpt_prompt=True,
        mask_prompt=True,
        box_prompt=True,
        smpl_head_depth=2,
        smpl_head_hidden_dim=256,
        
        # Body model settings
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True),
        
        # Camera settings
        cam_encoder=None,
        
        # Other settings
        train=True,
        inference=False,
        focal_length=[5000., 5000.],
        rotation_matrix=False,
        init_pose=True,

        FOV=pi/3,
        num_poses=24,
        dim_shape=10,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.train_mode = train
        self.inference = inference
        
        # SAM2 Image Encoder
        self.image_encoder = SAM2ImageEncoder(sam2_model_cfg, sam2_checkpoint, enable_video)
        
        # PHMR Prompt Encoder (SAM2 특징과 통합)
        self.prompt_encoder = SAM2PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=(img_size, img_size),
            text_prompt=text_prompt,
            kpt_prompt=kpt_prompt,
            mask_prompt=mask_prompt,
            box_prompt=box_prompt
        )
        
        # PHMR SMPL Decoder
        self.smplx_decoder = SAM2SMPLDecoder(
            # transformer_dim=embed_dim,
            transformer_dim=embed_dim,
            smpl_head_depth=smpl_head_depth,
            smpl_head_hidden_dim=smpl_head_hidden_dim,
            is_init_pose=init_pose,
        )
        
        # Camera Encoder (PHMR 스타일)
        if cam_encoder is not None:
            self.cam_encoder = cam_encoder
        else:
            self.cam_encoder = CameraEncoder(embed_dim)
        
        # Body model initialization
        # self.body_model = build_body_model(body_model)
        # for param in self.body_model.parameters():
        #     param.requires_grad = False

        self.input_size = input_size
        self.focal = input_size/(2*tan(FOV/2))
        self.FOV = FOV
        cam_intrinsics = torch.tensor([[self.focal,0.,self.input_size/2],
                                            [0.,self.focal,self.input_size/2],
                                            [0.,0.,1.]])
        self.register_buffer('cam_intrinsics', cam_intrinsics)
        # human model
        self.num_poses = num_poses
        self.dim_shape = dim_shape
        self.human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        # self.human_model = smpl_gendered
        # init params (following multi-hmr)
        smpl_mean_params = np.load(smpl_mean_path, allow_pickle = True)
        self.register_buffer('mean_pose', torch.from_numpy(smpl_mean_params['pose']))
        self.register_buffer('mean_shape', torch.from_numpy(smpl_mean_params['shape']))
        
        # Other settings
        self.focal_length = focal_length
        self.rotation_matrix = rotation_matrix

    def generate_camera_intrinsics(self, img_h, img_w, focal_length=None):
        """Generate camera intrinsics matrix for PHMR compatibility"""
        if focal_length is None:
            focal_length = self.focal_length[0] if hasattr(self, 'focal_length') else 5000.0
        
        if isinstance(focal_length, (list, tuple)):
            fx, fy = focal_length[0], focal_length[1]
        else:
            fx = fy = focal_length
        
        # Center of image
        cx, cy = img_w / 2.0, img_h / 2.0
        
        # Create 3x3 intrinsic matrix
        cam_int = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return cam_int

    
    # def forward(self, data_batch, sam2_feats=None, targets=None, **kwargs):
    def forward(self, samples, targets, sat_use_gt = False, detach_j3ds = False, sam2_feats=None):
        """Forward pass - Image model compatible with engine.py"""
        
        # Data preparation
        # if isinstance(data_batch, dict):
        #     samples, targets = self.prepare_targets(data_batch)
        # elif isinstance(data_batch, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(data_batch)
        # else:
        #     samples = data_batch

        # samples = [self.unnormalize(img_norm).astype("uint8") for img_norm in samples]
        # cv2.imwrite("output.jpg", cv2.cvtColor(samples[1].cpu().permute(1,2,0).numpy().astype("uint8"), cv2.COLOR_RGB2BGR))
        
        device = samples[0].device
        batch_size = len(samples)
        # # SAM2 feature extraction (if sam2_feats not provided)
        if sam2_feats is None:
            # sam2_feats, sam2_boxes, sam2_masks = self.image_encoder(
            sam2_feats, all_sparse_embeddings, all_dense_embeddings = self.image_encoder(
                samples, 
                [target['boxes'] for target in targets],
                [target['img_size'] for target in targets],
                device,
                targets,
                getattr(self, "detector", None),
                input_size=self.input_size,
            )
        else:
            sam2_boxes, sam2_masks = [], []
        
        # Image model processing - batch processing instead of per-image
        # batch_size = len(samples.tensors)
        
        # Use camera intrinsics from prepare_targets or generate if not available
        # if 'cam_int' in data_batch:
        #     cam_int = data_batch['cam_int']
        # else:
        # Generate camera intrinsics for PHMR compatibility
        cam_int_list = []
        for i in range(batch_size):
            if targets[i]['cam_intrinsics'].shape[0] != 1:
                img_h, img_w = targets[i]['img_size']
                focal_length = targets[i]['focals'][0]
                cam_int_i = self.generate_camera_intrinsics(img_h, img_w, focal_length)
                cam_int_i = cam_int_i.unsqueeze(0).to(device)  # Add batch dimension and move to device
            else:
                cam_int_list.append(targets[i]['cam_intrinsics'])
        cam_int = torch.cat(cam_int_list, dim=0)
        
        # SAM2 features to PHMR format conversion (batch processing)
        # image_embeddings, sparse_embeddings, dense_embeddings = self.prompt_encoder(
        image_embeddings = self.prompt_encoder(sam2_feats)
        
        # Camera encoding (batch processing)
        image_embeddings = self.cam_encoder(image_embeddings, cam_int)
        
        # Dense positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()

        img_size = torch.stack([t['img_size'].flip(0) for t in targets])
        # img_size = torch.stack([t['img_size'] for t in targets])
        valid_ratio = img_size/self.input_size
        
        # cam_intrinsics = self.cam_intrinsics.repeat(batch_size, 1, 1, 1)
        # cam_intrinsics[...,:2,2] = cam_intrinsics[...,:2,2] * valid_ratio[:, None, :]
        
        outputs_poses = []
        outputs_shapes = []
        outputs_verts = []
        outputs_j3ds = []
        outputs_j2ds = []
        outputs_depths = []
        outputs_transl = []
        # SMPLX prediction (batch processing)
        for _cam_int, _image_embeddings, _image_pe, sparse_embeddings, dense_embeddings \
            in zip(cam_int, image_embeddings, image_pe.repeat(cam_int.shape[0], 1, 1, 1), all_sparse_embeddings, all_dense_embeddings):
            # (pose, shape, transl, transl_c, depth_c, 
            # left_hand_pose, right_hand_pose, jaw_pose, 
            # expression, features) = self.smplx_decoder(
            #     _cam_int, _image_embeddings, _image_pe, sparse_embeddings, 
            #     dense_embeddings, crossperson=False
            # )
            # (pose, shape, transl, transl_c, depth_c, features) = self.smplx_decoder(
            (pose, shape, cam_xys, features) = self.smplx_decoder(
                _cam_int, _image_embeddings, _image_pe, sparse_embeddings, 
                dense_embeddings, crossperson=False
            )

            output = {
                'pose': pose,
                'betas': shape,
                'cam_xys': cam_xys,
                'cam_int': _cam_int.unsqueeze(0),
            }

            verts_cam, j3ds_cam, j2ds_img, depths, transl, outputs_pose, outputs_shape = self.process_output(output, use_mean_hands=False)
        
            outputs_poses.append(outputs_pose)
            outputs_shapes.append(outputs_shape)
            outputs_verts.append(verts_cam)
            outputs_j3ds.append(j3ds_cam)
            outputs_j2ds.append(j2ds_img)
            outputs_depths.append(depths)
            outputs_transl.append(transl)
        # pred_intrinsics = cam_intrinsics
        pred_intrinsics = cam_int

        out = {'pred_poses': outputs_poses, 'pred_betas': outputs_shapes,
               'pred_j3ds': outputs_j3ds, 'pred_j2ds': outputs_j2ds,
               'pred_verts': outputs_verts, 'pred_intrinsics': pred_intrinsics,
               'pred_depths': outputs_depths, 'pred_transl': outputs_transl,
               'img': samples}
        
        return out

    def process_output(self, output, use_mean_hands=False, detach_j3ds=False):
        """PHMR-style output post-processing using self.body_model"""
        # from detrsmpl.utils.geometry import rot6d_to_rotmat
        # from detrsmpl.utils.transforms import rotmat_to_aa
        # from detrsmpl.utils.geometry import batch_rodrigues
        
        outputs_pose_6d = self.mean_pose.view(1, -1)
        outputs_shape = self.mean_shape.view(1, -1)
        K = output['cam_int']
        cam_xys = output['cam_xys'].reshape(-1, 3) 
        # shape = outputs_shape + output['betas'].reshape(-1, 10)
        # pose = output['pose'].reshape(-1, 53, 6)  # SMPL-X has 53 joints
        # pose = outputs_pose_6d + output['pose'].reshape(-1, 24 * 6)  # SMPL-X has 53 joints
        shape = output['betas'].reshape(-1, 10)
        pose = output['pose'].reshape(-1, 24 * 6)  # SMPL-X has 53 joints
        # if self.rotation_matrix:
        pose = rot6d_to_axis_angle(pose.unsqueeze(0))
        # else:
        # rotmat = rot6d_to_rotmat(pose)
        bs, num_queries, _ = pose.shape
        
        pose = pose.flatten(0,1) # (bs*n_q,24*3)
        verts, joints = self.human_model(poses=pose, betas=shape)
    
        num_verts = verts.shape[1]
        num_joints = joints.shape[1]
        verts = verts.reshape(num_queries,num_verts,3)
        joints = joints.reshape(num_queries,num_joints,3)

        scale = 2*cam_xys[:,2:].sigmoid() + 1e-6
        t_xy = cam_xys[:,:2]/scale
        t_z = (2*self.focal)/(scale*self.input_size)    # (bs,num_queries,1)
        transl = torch.cat([t_xy,t_z],dim=-1)[:,None,:]    # (bs,nq,1,3)

        verts_cam = verts + transl # only for visualization and evaluation
        j3ds_cam = joints + transl

        if detach_j3ds:
            j2ds_homo = torch.matmul(joints.detach() + transl, output['cam_int'].transpose(-1, -2))
        else:
            j2ds_homo = torch.matmul(j3ds_cam, output['cam_int'].transpose(-1, -2))
        j2ds_img = (j2ds_homo[..., :2] / (j2ds_homo[..., 2, None] + 1e-6)).reshape(num_queries,num_joints,2)

        depths = j3ds_cam[:,0,2:]   # (bs, n_q, 1)
        depths = torch.cat([depths, depths/self.focal], dim=-1) # (bs, n_q, 2)
        return verts_cam, j3ds_cam, j2ds_img, depths, transl.flatten(2), pose, shape


def build_hmr_model(args, set_criterion=True):
    model = Sam2Model(args.input_size)

    if set_criterion:
        weight_dict = args.weight_dict
        losses = args.losses

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{k}.{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        criterion = SetCriterionSAM2(weight_dict, losses = losses, j2ds_norm_scale = args.input_size)
        return model, criterion
    else:
        return model, None
