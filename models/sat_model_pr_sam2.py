# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import os

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
from .criterion import SetCriterion_SATPR, SetCriterion, SetCriterion_SATPR_IMG
from .dn_components import prepare_for_cdn, dn_post_process, prepare_prompt_query_box
import copy

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer
from models.prompt.prompt_encoder import PromptEncoderBox

from PIL import Image

from models.sam2.build_sam2 import build_sam2

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, j2ds_to_bboxes_xywh
from typing import List, Dict, Any, Tuple, Optional
import hydra


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prompt.prompt_encoder import PromptEncoderBox
from inspect import isfunction
from einops import rearrange
from models.encoders.dinov2.layers import MemEffCrossAttention, SelfAttention, MemEffCrossAttentionWeight, CrossAttention
from models.utils import MaskEncoder
from scipy.optimize import linear_sum_assignment 
from collections import defaultdict



def exists(val):
    "Code modified from https://github.com/shubham-goel/4D-Humans/blob/a0def798c7eac811a63c8220fcc22d983b39785e/hmr2/models/components/pose_transformer.py#L17"
    return val is not None

def default(val, d):
    "Code modified from https://github.com/shubham-goel/4D-Humans/blob/a0def798c7eac811a63c8220fcc22d983b39785e/hmr2/models/components/pose_transformer.py#L21"
    if exists(val):
        return val
    return d() if isfunction(d) else d

def pairwise_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    # [N,1,4] vs [1,M,4]
    b1 = boxes1[:, None, :]  # (N,1,4)
    b2 = boxes2[None, :, :]  # (1,M,4)

    inter_x1 = torch.maximum(b1[...,0], b2[...,0])
    inter_y1 = torch.maximum(b1[...,1], b2[...,1])
    inter_x2 = torch.minimum(b1[...,2], b2[...,2])
    inter_y2 = torch.minimum(b1[...,3], b2[...,3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter   = inter_w * inter_h

    area1 = (boxes1[:,2] - boxes1[:,0]).clamp(min=0) * (boxes1[:,3] - boxes1[:,1]).clamp(min=0)  # (N,)
    area2 = (boxes2[:,2] - boxes2[:,0]).clamp(min=0) * (boxes2[:,3] - boxes2[:,1]).clamp(min=0)  # (M,)
    union = area1[:,None] + area2[None,:] - inter

    iou = torch.zeros_like(union, dtype=torch.float32)
    valid = union > 0
    iou[valid] = inter[valid] / union[valid]
    return iou

def get_similarity(mk: torch.Tensor,
                   ms: torch.Tensor,
                   qk: torch.Tensor,
                   qe: torch.Tensor,
                   add_batch_dim=False) -> torch.Tensor:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)

    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See XMem's appendix for derivation
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq + two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)  # B*N*HW

    return similarity




class VideoModel(nn.Module):
    """ One-stage Multi-person Human Mesh Estimation via Scale-adaptive Tokens """
    def __init__(self, encoder, decoder,
                    num_queries,
                    input_size,
                    sat_cfg = {'use_sat': False},
                    dn_cfg = {'use_dn': False},
                    train_pos_embed = True,
                    aux_loss=True, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=True,
                    random_refpoints_xy=False,
                    num_poses=24,
                    dim_shape=10,
                    FOV=pi/3,
                    memory_update_type='sim',
                    memory_update_early=False,
                    ):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See ./encoders.
            decoder: torch module of the decoder architecture. See decoder.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()

        # ========== Start of common settings =============
        self.input_size = input_size
        hidden_dim = decoder.d_model
        num_dec_layers = decoder.dec_layers
        self.hidden_dim = hidden_dim
        # camera model
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
        # init params (following multi-hmr)
        smpl_mean_params = np.load(smpl_mean_path, allow_pickle = True)
        self.register_buffer('mean_pose', torch.from_numpy(smpl_mean_params['pose']))
        self.register_buffer('mean_shape', torch.from_numpy(smpl_mean_params['shape']))
        # ========== End of common settings =============


        # ========== Start of SAT-encoder settings =============
        self.encoder = encoder
        
        self.patch_size = encoder.patch_size
        assert self.patch_size == 14
        
        self.use_sat = sat_cfg['use_sat']
        self.sat_cfg = sat_cfg

        if self.use_sat:
            assert sat_cfg['num_lvls'] >= 2
            assert self.input_size % (self.patch_size<<2) == 0

            self.feature_size = []
            for lvl in range(sat_cfg['num_lvls']):
                patch_size = self.patch_size<<lvl
                self.feature_size.append(self.input_size / patch_size)

            # build z_order curve
            z_depth = math.ceil(math.log2(self.feature_size[1]))
            z_map, ys, xs = build_z_map(z_depth)
            self.register_buffer('z_order_map', z_map)
            self.register_buffer('y_coords', ys)
            self.register_buffer('x_coords', xs)

            self.enc_inter_norm = copy.deepcopy(encoder.norm)
            self.scale_head = MLP(encoder.embed_dim, encoder.embed_dim, 2, 4)
            self.encoder_patch_proj = _get_clones(encoder.patch_embed.proj, 2)
            self.encoder_patch_norm = _get_clones(encoder.patch_embed.norm, 2)

            if sat_cfg['lvl_embed']:
                # same as level_embed in Deformable-DETR
                self.level_embed = nn.Parameter(torch.Tensor(sat_cfg['num_lvls'],hidden_dim))
                nn.init.normal_(self.level_embed)
        else:
            assert self.input_size % self.patch_size == 0
            self.feature_size = [self.input_size // self.patch_size]
            self.encoder_patch_proj = copy.deepcopy(encoder.patch_embed.proj)
            self.encoder_patch_norm = copy.deepcopy(encoder.patch_embed.norm)

        # cls_token and register tokens
        encoder_cr_token = self.encoder.cls_token.view(1,-1) + self.encoder.pos_embed.float()[:,0].view(1,-1)
        if self.encoder.register_tokens is not None:
            encoder_cr_token = torch.cat([encoder_cr_token, self.encoder.register_tokens.view(self.encoder.num_register_tokens,-1)], dim=0)
        self.encoder_cr_token = nn.Parameter(encoder_cr_token)
        
        self.encoder_pos_embeds = nn.Parameter(self.encoder.interpolate_pos_encoding3(self.feature_size[0]).detach())
        if not train_pos_embed:
            self.encoder_pos_embeds.requires_grad = False
        
        self.preprocessed_pos_lvl1 = None
        
        # delete unwanted params
        del(self.encoder.mask_token)
        del(self.encoder.pos_embed)
        del(self.encoder.patch_embed)
        del(self.encoder.cls_token)
        del(self.encoder.register_tokens)
        # ========== End of SAT-encoder settings =============


        
        # ========== Start of decoder settings =============
        self.num_queries = num_queries
        self.decoder = decoder
        
        # embed_dim between encoder and decoder can be different
        self.feature_proj = nn.Linear(encoder.embed_dim, hidden_dim)

        # bbox
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # poses (use 6D rotation)
        self.pose_head = MLP(hidden_dim, hidden_dim, num_poses*6, 6)
        # shape
        self.shape_head = MLP(hidden_dim, hidden_dim, dim_shape, 5)
        # cam_trans
        self.cam_head = MLP(hidden_dim, hidden_dim//2, 3, 3)
        # confidence score
        self.conf_head = nn.Linear(hidden_dim, 1)
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.conf_head.bias.data = torch.ones(1) * bias_value

        # for iter update
        self.pose_head = _get_clones(self.pose_head, num_dec_layers)
        self.shape_head = _get_clones(self.shape_head, num_dec_layers)
        
        # setting query dim (bboxes as queries)
        self.query_dim = query_dim
        assert query_dim == 4
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.aux_loss = aux_loss
        self.iter_update = iter_update
        assert iter_update
        if self.iter_update:
            self.decoder.decoder.bbox_embed = self.bbox_embed

        assert bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # ========== End of decoder settings =============


        # for dn training
        self.use_dn = dn_cfg['use_dn']
        self.dn_cfg = dn_cfg
        if self.use_dn:
            assert dn_cfg['dn_number'] > 0
            if dn_cfg['tgt_embed_type'] == 'labels':
                self.dn_enc = nn.Embedding(dn_cfg['dn_labelbook_size'], hidden_dim)
            elif dn_cfg['tgt_embed_type'] == 'params':
                self.dn_enc = nn.Linear(num_poses*3 + dim_shape, hidden_dim)
            elif dn_cfg['tgt_embed_type'] == 'boxes':
                self.dn_enc = nn.Linear(num_poses*3 + dim_shape, hidden_dim)
                self.dn_boxes_enc = nn.Linear(hidden_dim*3, hidden_dim)
            else:
                raise NotImplementedError

        # self.prompt_encoder = PromptEncoderBox(
        #     embed_dim=hidden_dim,
        #     input_image_size=[self.input_size/14, self.input_size/14],
        # )
        # self.dn_prompt_enc = nn.Linear(hidden_dim * 3, hidden_dim)
        # # self.dn_prompt_enc = nn.Linear(num_poses*3 + dim_shape, hidden_dim)
        self.prompt_cross_attn = MemEffCrossAttentionWeight(4)

        self.hidden_dim = hidden_dim
        self.embed_dim = hidden_dim
        self.img_size = 1288

        # self.next_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # with torch.no_grad(): 
        #     for p_new, p_old in zip(self.next_bbox_embed.parameters(), self.bbox_embed[-1].parameters()):
        #         p_new.copy_(p_old) 

        # for p in self.next_bbox_embed.parameters():
        #     p.requires_grad_(True)

        self.ema_factor = 0.8

        self.memory_update_type = memory_update_type
        if self.memory_update_type != "sim":
            self.tracking_query_cross_attn = CrossAttention(hidden_dim)
        self.memory_update_early = memory_update_early
        
    def lvl_pooling(self, tokens):
        assert len(tokens)%4 == 0
        C = tokens.shape[-1]
        return torch.max(tokens.view(-1, 4, C), dim=1)[0]
                
    def get_scale_map(self, x_list):
        if self.sat_cfg['use_additional_blocks']:
            x_list = self.encoder.forward_additional_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
        else:
            x_list = self.encoder.forward_specific_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
        
        cr_token_list = [x[:, :1 + self.encoder.num_register_tokens, :].squeeze(0) for x in x_list]
        x_tokens = torch.cat([x[:, 1 + self.encoder.num_register_tokens:, :].squeeze(0) for x in x_list], dim=0)
        scale_map = self.scale_head(self.enc_inter_norm(x_tokens)).sigmoid()
        return scale_map, cr_token_list, x_tokens

    def pad_mask(self, mask):
        mask = mask.reshape(-1,4)
        mask[torch.any(mask, dim=1)] = True
        return mask.flatten()

    def forward_encoder(self, samples, targets, masks=None, use_gt = False):
        B = len(samples)
        C = self.encoder.embed_dim
        cr_token_list = [self.encoder_cr_token]*len(samples)

        if not self.use_sat:
            # img2token
            lvl0_feature_hw = [(img.shape[1]//self.patch_size, img.shape[2]//self.patch_size) for img in samples]
            lvl0_token_lens = [h*w for (h,w) in lvl0_feature_hw]
            lvl0_img_patches = torch.cat([img2patch_flat(img, patch_size = self.patch_size)\
                                        for img in samples], dim=0)
            lvl0_tokens = self.encoder_patch_norm(self.encoder_patch_proj(lvl0_img_patches).flatten(1))     

            # token position information
            full_grids = torch.meshgrid(torch.arange(self.feature_size[0]), torch.arange(self.feature_size[0]), indexing='ij')
            lvl0_pos_y = torch.cat([full_grids[0][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)
            lvl0_pos_x = torch.cat([full_grids[1][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)

            # pos_embed
            full_pos_embed = self.encoder_pos_embeds
            lvl0_pos_embed = torch.cat([full_pos_embed[:h,:w].flatten(0,1)\
                                        for (h,w) in lvl0_feature_hw], dim=0)
            lvl0_tokens = lvl0_tokens + lvl0_pos_embed

            # convert to list for DINOv2 input
            x_list = [torch.cat([cr, lvl0],dim=0).unsqueeze(0)\
                                for (cr, lvl0) \
                                in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))]
            
            
            lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
            lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
            pos_y_list = list(lvl0_pos_y_norm.split(lvl0_token_lens))
            pos_x_list = list(lvl0_pos_x_norm.split(lvl0_token_lens))
            scale_map_dict = None
            # also create lvl_list for patch visualization
            lvl_list = [torch.zeros_like(pos,dtype=int) for pos in pos_x_list]

        else:
            lvl1_feature_hw = [(img.shape[1]//(2*self.patch_size), img.shape[2]//(2*self.patch_size)) for img in samples]
            lvl1_token_lens = [h*w for (h,w) in lvl1_feature_hw]

            lvl1_img_patches_28, lvl1_zorders = [], []
            lvl1_pos_y, lvl1_pos_x = [], []
            lvl1_bids = []

            for i, img in enumerate(samples):
                z_patches, z_order, pos_y, pos_x = to_zorder(img2patch(img, patch_size = 2*self.patch_size), 
                                                             z_order_map = self.z_order_map,
                                                             y_coords = self.y_coords,
                                                             x_coords = self.x_coords)

                lvl1_img_patches_28.append(z_patches)
                
                lvl1_zorders.append(z_order)
                lvl1_pos_y.append(pos_y)
                lvl1_pos_x.append(pos_x)
                lvl1_bids.append(torch.full_like(pos_y, i, dtype=torch.int64))
            

            
            lvl1_img_patches_28 = torch.cat(lvl1_img_patches_28, dim=0)
            lvl1_zorders = torch.cat(lvl1_zorders, dim=0)
            lvl1_pos_y = torch.cat(lvl1_pos_y, dim=0)
            lvl1_pos_x = torch.cat(lvl1_pos_x, dim=0)
            lvl1_bids = torch.cat(lvl1_bids, dim=0)

            

            # (L1, 3, 28, 28)
            assert len(lvl1_img_patches_28) == sum(lvl1_token_lens)
            lvl1_img_patches = F.interpolate(lvl1_img_patches_28, size = (14,14), mode='bilinear', align_corners=False)
            # (L1, 3, 14, 14)
            lvl1_tokens = self.encoder_patch_norm[1](self.encoder_patch_proj[1](lvl1_img_patches).flatten(1))
            # (L1, C)
            
            
            
            assert len(lvl1_pos_y) == len(lvl1_tokens)
            full_pos_embed = self.preprocessed_pos_lvl1 if not self.training\
                                else F.interpolate(self.encoder_pos_embeds.unsqueeze(0).permute(0, 3, 1, 2),
                                            mode="bicubic",
                                            antialias=self.encoder.interpolate_antialias,
                                            size = (int(self.feature_size[1]),int(self.feature_size[1]))).squeeze(0).permute(1,2,0)
            lvl1_pos_embed = torch.cat([full_pos_embed[ys,xs]\
                                        for (ys,xs) in zip(lvl1_pos_y.split(lvl1_token_lens), lvl1_pos_x.split(lvl1_token_lens))], dim=0)
            lvl1_tokens = lvl1_tokens + lvl1_pos_embed

            # get scale map (flattened)
            x_list = [torch.cat([cr, lvl1],dim=0).unsqueeze(0)\
                                 for (cr, lvl1) \
                                 in zip(cr_token_list, lvl1_tokens.split(lvl1_token_lens))]
            scale_map, updated_cr_list, updated_lvl1 = self.get_scale_map(x_list)
            # for visualization
            scale_map_dict = {'scale_map': scale_map, 'lens': lvl1_token_lens, 'hw': lvl1_feature_hw,
                              'pos_y': lvl1_pos_y, 'pos_x': lvl1_pos_x}
            
            # get sat masks
            conf_thresh = self.sat_cfg['conf_thresh']
            scale_thresh = self.sat_cfg['scale_thresh']
            if use_gt:
                scale_map = torch.cat([tgt['scale_map'].view(-1,2) for tgt in targets], dim=0)

            lvl1_valid_mask = scale_map[:,0] > conf_thresh
            lvl1_sat_mask = lvl1_valid_mask & (scale_map[:,1] < scale_thresh)
        
            # prepare sat tokens (lvl0)
            lvl0_token_lens = [msk.sum().item()<<2 for msk in lvl1_sat_mask.split(lvl1_token_lens)]
            lvl1_sat_patches_28 = lvl1_img_patches_28[lvl1_sat_mask] # (L0//4, 3, 28, 28)            
            lvl0_tokens = self.encoder_patch_norm[0](self.encoder_patch_proj[0](lvl1_sat_patches_28).permute(0, 2, 3, 1).flatten(0,2))

            assert len(lvl0_tokens) == sum(lvl0_token_lens)
            # lvl0 positions
            lvl0_pos_y, lvl0_pos_x = lvl1_pos_y[lvl1_sat_mask], lvl1_pos_x[lvl1_sat_mask]
            lvl0_pos_y = (lvl0_pos_y<<1)[:,None].repeat(1,4).flatten()
            lvl0_pos_x = (lvl0_pos_x<<1)[:,None].repeat(1,4).flatten()
            lvl0_pos_y[2::4] += 1
            lvl0_pos_y[3::4] += 1
            lvl0_pos_x[1::2] += 1
            assert len(lvl0_pos_x) == len(lvl0_tokens)
                     
            # lvl0 pos_embed
            full_pos_embed = self.encoder_pos_embeds
            lvl0_pos_embed = torch.cat([full_pos_embed[ys,xs]\
                                        for (ys,xs) in zip(lvl0_pos_y.split(lvl0_token_lens), lvl0_pos_x.split(lvl0_token_lens))], dim=0)
            lvl0_tokens = lvl0_tokens + lvl0_pos_embed


            # update tokens
            x_list = [torch.cat([cr, lvl0],dim=0).unsqueeze(0)\
                            for (cr, lvl0) \
                            in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))]
            x_list = self.encoder.forward_specific_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
            lvl0_tokens = torch.cat([x[:, 1 + self.encoder.num_register_tokens:, :].squeeze(0) for x in x_list], dim=0)
            assert len(lvl0_pos_x) == len(lvl0_tokens)
            # also update lvl1 and crs
            lvl1_tokens = updated_lvl1
            cr_token_list = updated_cr_list

            

            if self.sat_cfg['num_lvls'] == 2:
                # drop corresponding lvl1 tokens
                lvl1_keep = ~lvl1_sat_mask
                lvl1_token_lens = [msk.sum().item() for msk in lvl1_keep.split(lvl1_token_lens)]
                lvl1_tokens = lvl1_tokens[lvl1_keep]
                lvl1_pos_y = lvl1_pos_y[lvl1_keep]
                lvl1_pos_x = lvl1_pos_x[lvl1_keep]

                # normalize positions
                lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                lvl1_pos_y_norm = (lvl1_pos_y.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]
                lvl1_pos_x_norm = (lvl1_pos_x.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]

                # merge all
                x_list = [torch.cat([cr, lvl0, lvl1]).unsqueeze(0) \
                                for cr, lvl0, lvl1 \
                                in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens), lvl1_tokens.split(lvl1_token_lens))]
                pos_y_list = [torch.cat([lvl0, lvl1]) \
                                    for lvl0, lvl1 \
                                    in zip(lvl0_pos_y_norm.split(lvl0_token_lens), lvl1_pos_y_norm.split(lvl1_token_lens))]
                pos_x_list = [torch.cat([lvl0, lvl1]) \
                                    for lvl0, lvl1 \
                                    in zip(lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens))]
                lvl_list = [torch.cat([torch.zeros_like(lvl0, dtype=int), torch.ones_like(lvl1, dtype=int)]) \
                                    for lvl0, lvl1 \
                                    in zip(lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens))]


            else:
                # prune lvl1 correspond to lvl0
                try:
                    lvl1_valid_mask = self.pad_mask(lvl1_valid_mask)
                except:
                    import pdb; pdb.set_trace()
                lvl1_keep = lvl1_valid_mask & (~lvl1_sat_mask)
                lvl1_to_lvl2 = ~lvl1_valid_mask

                token_lvls = [lvl0_tokens, lvl1_tokens]
                token_lens_lvls = [lvl0_token_lens, lvl1_token_lens]
                pos_y_lvls = [lvl0_pos_y, lvl1_pos_y]
                pos_x_lvls = [lvl0_pos_x, lvl1_pos_x]

                to_next_lvl = lvl1_to_lvl2
                keep = lvl1_keep
                lvl_zorders = lvl1_zorders
                lvl_bids = lvl1_bids
                pad_vals = torch.full((3,), -1, dtype=lvl_zorders.dtype, device=lvl_zorders.device)
                for lvl in range(self.sat_cfg['num_lvls']-2):
                    # if to_next_lvl.sum() == 0:
                    #     break
                    next_tokens = self.lvl_pooling(token_lvls[-1][to_next_lvl])
                    # next_tokens = torch.max(token_lvls[-1][to_next_lvl].view(-1,4,C), dim=1)[0]
                    next_pos_y = pos_y_lvls[-1][to_next_lvl][::4]>>1
                    next_pos_x = pos_x_lvls[-1][to_next_lvl][::4]>>1
                    next_lens = [msk.sum().item()//4 for msk in to_next_lvl.split(token_lens_lvls[-1])]

                    
                    token_lvls[-1] = token_lvls[-1][keep]
                    pos_y_lvls[-1] = pos_y_lvls[-1][keep]
                    pos_x_lvls[-1] = pos_x_lvls[-1][keep]
                    token_lens_lvls[-1] = [msk.sum().item() for msk in keep.split(token_lens_lvls[-1])]
                    
                    token_lvls.append(next_tokens)
                    token_lens_lvls.append(next_lens)
                    pos_y_lvls.append(next_pos_y)
                    pos_x_lvls.append(next_pos_x)
                    
                    if lvl < self.sat_cfg['num_lvls']-3:
                        lvl_zorders = lvl_zorders[to_next_lvl][::4]>>2
                        lvl_bids = lvl_bids[to_next_lvl][::4]

                        z_starts_idx = torch.where((lvl_zorders&3)==0)[0]
                        padded_z = torch.cat([lvl_zorders, pad_vals])
                        padded_bids = torch.cat([lvl_bids, pad_vals])
                        valids = (padded_z[z_starts_idx] + 3 == padded_z[z_starts_idx + 3]) & (padded_bids[z_starts_idx] == padded_bids[z_starts_idx + 3])
                        valid_starts = z_starts_idx[valids]
                    
                        to_next_lvl = torch.zeros_like(lvl_zorders, dtype=bool)
                        to_next_lvl[valid_starts] = True
                        to_next_lvl[valid_starts+1] = True
                        to_next_lvl[valid_starts+2] = True
                        to_next_lvl[valid_starts+3] = True

                        keep = ~to_next_lvl

                norm_pos_y_lvls = [(pos_y.to(dtype=lvl0_tokens.dtype) + 0.5)/self.feature_size[i]  for i, pos_y in enumerate(pos_y_lvls)]
                norm_pos_x_lvls = [(pos_x.to(dtype=lvl0_tokens.dtype) + 0.5)/self.feature_size[i]  for i, pos_x in enumerate(pos_x_lvls)]

                x_list = [torch.cat([cr, *lvls]).unsqueeze(0) \
                                for cr, *lvls \
                                in zip(cr_token_list, *[tokens.split(lens) for (tokens, lens) in zip(token_lvls, token_lens_lvls)])]
                pos_y_list = [torch.cat([*lvls]) \
                                    for lvls \
                                    in zip(*[pos_y.split(lens) for (pos_y, lens) in zip(norm_pos_y_lvls, token_lens_lvls)])]
                pos_x_list = [torch.cat([*lvls]) \
                                    for lvls \
                                    in zip(*[pos_x.split(lens) for (pos_x, lens) in zip(norm_pos_x_lvls, token_lens_lvls)])]
                lvl_list = [torch.cat([torch.full_like(lvl, i, dtype=torch.int64) for i, lvl in enumerate(lvls)]) \
                                    for lvls \
                                    in zip(*[pos_x.split(lens) for (pos_x, lens) in zip(norm_pos_x_lvls, token_lens_lvls)])]

                

        start = self.sat_cfg['get_map_layer'] if self.use_sat else 0
        _, final_feature_list = self.encoder.forward_specific_layers_list(x_list, start = start, norm=True)

        # default dab-detr pipeline
        # proj
        token_lens = [feature.shape[1] for feature in final_feature_list]
        final_features = self.feature_proj(torch.cat(final_feature_list,dim=1).squeeze(0)) # (sum(L), C)
        assert tuple(final_features.shape) == (sum(token_lens), self.hidden_dim)
        # positional encoding
        pos_embeds = position_encoding_xy(torch.cat(pos_x_list,dim=0), torch.cat(pos_y_list,dim=0), embedding_dim=self.hidden_dim)
        if self.use_sat and self.sat_cfg['lvl_embed']:
            lvl_embeds = self.level_embed[torch.cat(lvl_list,dim=0)]
            pos_embeds = pos_embeds + lvl_embeds

        sat_dict = {'pos_y': pos_y_list, 'pos_x': pos_x_list, 'lvl': lvl_list, 
                    #  'features': [feature.squeeze(0) for feature in final_feature_list],
                     'lens': token_lens}

        return final_features, pos_embeds, token_lens, scale_map_dict, sat_dict

    def process_smpl(self, poses, shapes, cam_xys, cam_intrinsics, detach_j3ds = False):
        bs, num_queries, _ = poses.shape # should be (bs,n_q,num_poses*3)

        # flatten and compute
        poses = poses.flatten(0,1) # (bs*n_q,24*3)
        shapes = shapes.flatten(0,1) # (bs*n_q,10)
        # with torch.no_grad():
        verts, joints = self.human_model(poses=poses,
                                        betas=shapes)
        num_verts = verts.shape[1]
        num_joints = joints.shape[1]
        verts = verts.reshape(bs,num_queries,num_verts,3)
        joints = joints.reshape(bs,num_queries,num_joints,3)

        # apply cam_trans and projection
        scale = 2*cam_xys[:,:,2:].sigmoid() + 1e-6
        t_xy = cam_xys[:,:,:2]/scale
        t_z = (2*self.focal)/(scale*self.input_size)    # (bs,num_queries,1)
        transl = torch.cat([t_xy,t_z],dim=2)[:,:,None,:]    # (bs,nq,1,3)

        verts_cam = verts + transl # only for visualization and evaluation
        j3ds_cam = joints + transl

        if detach_j3ds:
            j2ds_homo = torch.matmul(joints.detach() + transl, cam_intrinsics.transpose(2,3))
        else:
            j2ds_homo = torch.matmul(j3ds_cam, cam_intrinsics.transpose(2,3))
        j2ds_img = (j2ds_homo[..., :2] / (j2ds_homo[..., 2, None] + 1e-6)).reshape(bs,num_queries,num_joints,2)

        depths = j3ds_cam[:,:,0,2:]   # (bs, n_q, 1)
        depths = torch.cat([depths, depths/self.focal], dim=-1) # (bs, n_q, 2)

        return verts_cam, j3ds_cam, j2ds_img, depths, transl.flatten(2)


    def forward(self, samples: NestedTensor, targets, sam2_image_model=None, masks=None, sat_use_gt = False, detach_j3ds = False, boxes=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        assert isinstance(samples, (list, torch.Tensor))

        if self.training:
            self.preprocessed_pos_lvl1 = None

        elif self.preprocessed_pos_lvl1 is None and self.use_sat:
            self.preprocessed_pos_lvl1 = F.interpolate(self.encoder_pos_embeds.unsqueeze(0).permute(0, 3, 1, 2),
                                            mode="bicubic",
                                            antialias=self.encoder.interpolate_antialias,
                                            size = (int(self.feature_size[1]),int(self.feature_size[1]))).squeeze(0).permute(1,2,0)


        bs = len(targets)
        device = samples[0].device

        # get cam_intrinsics
        img_size = torch.stack([t['img_size'].flip(0) for t in targets])
        valid_ratio = img_size/self.input_size
        
        cam_intrinsics = self.cam_intrinsics.repeat(bs, 1, 1, 1)
        cam_intrinsics[...,:2,2] = cam_intrinsics[...,:2,2] * valid_ratio[:, None, :].to(cam_intrinsics.device)


        final_features, pos_embeds, token_lens, scale_map_dict, sat_dict\
             = self.forward_encoder(samples, targets, masks=masks, use_gt = sat_use_gt)

        # default dab-detr pipeline
        embedweight = (self.refpoint_embed.weight).unsqueeze(0).repeat(bs,1,1)
        tgt = (self.tgt_embed.weight).unsqueeze(0).repeat(bs,1,1)

        pr_points = None
        _boxes = []
        if 'init_boxes' in targets[0]:
            _boxes = targets[0]['init_boxes']
            if len(_boxes) !=0:
                imght, imgwidth = targets[0]['img_size']
                # imght, imgwidth = meta_data['img_size']
                # _boxes[...,[0,2]] = _boxes[...,[0,2]] / imgwidth
                # _boxes[...,[1,3]] = _boxes[...,[1,3]] / imght
                # print(_boxes)
                _boxes = box_xyxy_to_cxcywh(_boxes)/ self.input_size
                _boxes[...,2:] *= 1.2
                _boxes = box_cxcywh_to_xyxy(_boxes)
                _boxes[...,[0,2]] = _boxes[...,[0,2]].clamp(min=0.000001,max=(imgwidth-1)/self.input_size)
                _boxes[...,[1,3]] = _boxes[...,[1,3]].clamp(min=0.000001,max=(imght-1)/self.input_size)
                _boxes = box_xyxy_to_cxcywh(_boxes)
                # # print(_boxes)
                # # print([v['box'] for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age])
                # _boxes = torch.tensor([[0.5637, 0.3187, 0.8711, 0.2897]], device='cuda:0')

                max_len = max(b.shape[0] for b in _boxes)
                boxes = []
                for box in _boxes:
                    num_boxes = box.shape[0] 
                    num_repeats = (max_len + num_boxes - 1) // num_boxes 
                    repeated_tensor = box.repeat(num_repeats, 1)
                    sliced_tensor = repeated_tensor[:max_len]
                    boxes.append(sliced_tensor)
                pr_points = inverse_sigmoid(torch.stack(boxes)).to(device=embedweight.device, dtype=embedweight.dtype)

                attn_weights = self.prompt_cross_attn(q=pr_points, k=embedweight, v=embedweight)[:, 0, :, :]
                pr_tgt = torch.matmul(attn_weights, tgt)
                pr_tgt = pr_tgt[:, 0, :][None]
                pr_points = pr_points[:, 0, :][None]
        elif self.training or (hasattr(self, 'tracker') and len(self.tracker.tracks) != 0):
            if self.training:
                if self.img_seq_idx == 0:
                    _boxes = [t['boxes'] for t in targets]
                    _removed_boxes, _kept_indices = [], []
                    for boxes in _boxes:
                        n = len(boxes)
                        keep_idx = torch.tensor(
                            sorted(random.sample(range(n), k=max(1, int(n * (1 - random.uniform(0.1, 0.7)))))),
                            dtype=torch.long
                        )
                        _removed_boxes.append(boxes[keep_idx])
                        _kept_indices.append(keep_idx)
                    _boxes = _removed_boxes
                    self._kept_indices = _kept_indices
                else:
                    _removed_boxes = []
                    _boxes = [t['boxes'] for t in targets]
                    for boxes, keep_idx in zip(_boxes, self._kept_indices):
                        _removed_boxes.append(boxes[keep_idx])
                    _boxes = _removed_boxes

            elif (hasattr(self, 'tracker') and len(self.tracker.tracks) != 0):
                # _boxes = [box_xyxy_to_cxcywh((v['box'] * 0.5 + v['prev_box'] * 0.5) / self.input_size) for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age]
                # _boxes = [box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(v['box'] / self.input_size)) for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age]
                # prev_boxes = [box_xyxy_to_cxcywh(v['prev_box'] / self.input_size) for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age]
                
                # _boxes = [box_cxcywh_to_xyxy(v['kf'].get_xyxy() / self.input_size) for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age]
                # __boxes = [t['boxes'] for t in targets]
                # _boxes = list(__boxes[0].to(device=tgt.device).unbind(0))
                # print(_boxes)

                # _boxes = [box_xyxy_to_cxcywh(v['box']) / self.input_size for k, v in self.tracker.tracks.items() if v['age'] == 0 ]
                # print(_boxes)
                # _boxes = [
                #     (box_xyxy_to_cxcywh(v['box']) / self.input_size) * torch.tensor([1, 1, 1.2, 1.2], device=v['box'].device)
                #     for k, v in self.tracker.tracks.items()  if v['age'] == 0
                # ]
                _boxes = [v['box'] for k, v in self.tracker.tracks.items() if v['age'] == 0 ]
                if len(_boxes) !=0:
                    imght, imgwidth = targets[0]['img_size']
                    # imght, imgwidth = meta_data['img_size']
                    _boxes = box_xyxy_to_cxcywh(torch.stack(_boxes)) / self.input_size
                    _boxes[...,2:] *= 1.2
                    # print(2, _boxes)
                    _boxes = box_cxcywh_to_xyxy(_boxes)
                    _boxes[...,[0,2]] = _boxes[...,[0,2]].clamp(min=0.000001,max=(imgwidth-1)/self.input_size)
                    _boxes[...,[1,3]] = _boxes[...,[1,3]].clamp(min=0.000001,max=(imght-1)/self.input_size)
                    # print(3, _boxes)
                    _boxes = box_xyxy_to_cxcywh(_boxes)
                    # print(_boxes)
                    # print(4, _boxes)
                    # print([v['box'] for k, v in self.tracker.tracks.items() if v['age'] < self.tracker.max_age])

            pr_tgt = None
            if self.training:
                box_noise_scale = 0.2
                __boxes = []
                for pr_boxes in _boxes:
                    known_bboxs = pr_boxes
                    known_bbox_ = torch.zeros_like(known_bboxs)
                    known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
                    known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

                    diff = torch.zeros_like(known_bboxs)
                    diff[:, :2] = known_bboxs[:, 2:] / 2
                    diff[:, 2:] = known_bboxs[:, 2:] / 2

                    rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
                    rand_part = torch.rand_like(known_bboxs)
                    rand_part *= rand_sign
                    known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                            diff).to(device=device) * box_noise_scale
                    known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
                    known_bbox_expand = torch.zeros_like(known_bboxs)
                    known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
                    known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
                    __boxes.append(known_bbox_expand)
                _boxes = __boxes
            
            if len(_boxes) != 0:
                max_len = max(b.shape[0] for b in _boxes)
                boxes = []
                for box in _boxes:
                    num_boxes = box.shape[0] 
                    num_repeats = (max_len + num_boxes - 1) // num_boxes 
                    repeated_tensor = box.repeat(num_repeats, 1)
                    sliced_tensor = repeated_tensor[:max_len]
                    boxes.append(sliced_tensor)
                pr_points = inverse_sigmoid(torch.stack(boxes))

                attn_weights = self.prompt_cross_attn(q=pr_points, k=embedweight, v=embedweight)[:, 0, :, :]
                pr_tgt = torch.matmul(attn_weights, tgt)
                if (hasattr(self, 'tracker') and len(self.tracker.tracks) != 0):
                    pr_tgt = pr_tgt[:, 0, :][None]
                    pr_points = pr_points[:, 0, :][None]
        elif hasattr(self, 'infer_joint') and self.infer_joint:
            _boxes = [t['boxes'] for t in targets]
            if len(_boxes) != 0:
                max_len = max(b.shape[0] for b in _boxes)
                boxes = []
                for box in _boxes:
                    num_boxes = box.shape[0] 
                    num_repeats = (max_len + num_boxes - 1) // num_boxes 
                    repeated_tensor = box.repeat(num_repeats, 1)
                    sliced_tensor = repeated_tensor[:max_len]
                    boxes.append(sliced_tensor)
                pr_points = inverse_sigmoid(torch.stack(boxes)).to(embedweight.dtype)

                attn_weights = self.prompt_cross_attn(q=pr_points, k=embedweight, v=embedweight)[:, 0, :, :]
                pr_tgt = torch.matmul(attn_weights, tgt)

        if self.memory_update_early:
            if self.training:
                if self.img_seq_idx == 0:
                    self.query_bank.append(pr_tgt.clone().detach())
                else:
                    ori_queries = pr_tgt.clone()
                    self.query_bank.append(ori_queries.detach())
                    device = tgt.device
                    q_bank = torch.cat(self.query_bank, dim=1)
                    q_len = ori_queries.shape[1]
                    q_idx  = torch.arange(q_len, device=device).unsqueeze(1)          # (q,1)
                    kv_idx = (torch.arange(q_bank.shape[1], device=device) % q_len).unsqueeze(0)  # (1,kv)
                    kv_len = q_bank.size(1)       
                    bias = torch.where(q_idx == kv_idx, False, True)

                    if self.memory_update_type == "sim":
                        B, L, C = ori_queries.shape
                        _, S, _ = q_bank.shape
                        mk = q_bank.permute(0, 2, 1).contiguous()
                        qk = ori_queries.permute(0, 2, 1).contiguous()
                        qe = None

                        decay_rate = 0.8
                        kv_time_idx = torch.arange(kv_len, device=device) // q_len  # 0..T-1
                        time_decay = decay_rate ** (self.img_seq_idx+1 - 1 - kv_time_idx)            
                        ms = time_decay.view(B, 1, S).to(dtype=tgt.dtype, device=device) 
                        # ms = torch.ones(B, 1, S, device=device, dtype=tgt.dtype)
                        sim = get_similarity(mk=mk, ms=ms, qk=qk, qe=qe, add_batch_dim=False)
                        sim = sim.transpose(1, 2)  # (B, L, N)
                        sim = sim.masked_fill(bias[None], float('-inf'))

                        affinity = torch.softmax(sim, dim=-1) 

                        V = q_bank   # (1, S, C)
                        out = torch.einsum('bln,bnc->blc', affinity, V)

                        pr_tgt = out
                    else:
                        bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()
                        pr_tgt = ori_queries + self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias)
            else:
                if 'init_boxes' not in targets[0] and hasattr(self, 'tracker') and len(self.tracker.tracks) != 0 and pr_tgt is not None:
                    frame_buckets = defaultdict(list)
                    pr_pad = pr_tgt.shape[1]
                    query_bank = [v['prev_ori_query_list'] for k, v in self.tracker.tracks.items() if v['age'] == 0]
                    ori_queries = pr_tgt.clone()
                    device = tgt.device

                    for k, v in self.tracker.tracks.items():
                        if v['age'] != 0:
                            continue
                        # print(k)
                        for t, q in enumerate(v['prev_ori_query_list']):
                            if q is None:
                                continue
                            if isinstance(q, (list, tuple)):
                                q = torch.cat(q, dim=0)  # (n_ti, 768)
                            frame_buckets[t].append(q)   # (n_ti, 768)

                    ori_len = len(frame_buckets)
                    q_n = 0
                    for k, v in self.tracker.tracks.items():
                        if v['age'] == 0:
                            # print(k, q_n)
                            frame_buckets[ori_len].append(ori_queries[0, q_n].clone())
                            q_n += 1

                    ordered_ts = sorted(frame_buckets.keys())
                    frames = [torch.stack(frame_buckets[t]) for t in ordered_ts] 

                    device = frames[0].device
                    dtype  = frames[0].dtype
                    max_len = max(f.shape[0] for f in frames)
                    feat_dim = frames[0].shape[1]

                    valid_counts = []
                    padded_frames = []
                    for f in frames:
                        n = f.shape[0]
                        valid_counts.append(n)
                        if n < max_len:
                            pad = torch.zeros(max_len - n, feat_dim, device=device, dtype=dtype)
                            f = torch.cat([f, pad], dim=0)
                        padded_frames.append(f)  # (max_len, 768)

                    q_bank = torch.stack(padded_frames, dim=0)

                    q_bank = q_bank.reshape(1, -1, 768)
                    q_len = ori_queries.shape[1]
                    kv_len = q_bank.size(1)

                    q_idx  = torch.arange(q_len, device=device).unsqueeze(1)             # (L, 1)
                    kv_idx = (torch.arange(kv_len, device=device) % q_len).unsqueeze(0)  # (1, S)

                    bias = torch.where(q_idx == kv_idx, False, True)                         # (L_flat, kv)

                    valid_counts = torch.as_tensor(valid_counts, device=device)          # (T,)
                    kv_time_idx  = torch.arange(kv_len, device=device) // q_len          # (S,)
                    kv_valid     = (kv_idx.squeeze(0) < valid_counts[kv_time_idx])       # (S,)
                    bias = bias | (~kv_valid.unsqueeze(0))   

                    if self.memory_update_type == "sim":
                        B, L, C = ori_queries.shape
                        _, S, _ = q_bank.shape
                        mk = q_bank.permute(0, 2, 1).contiguous()
                        qk = ori_queries.permute(0, 2, 1).contiguous()
                        qe = None

                        decay_rate = 0.8
                        kv_time_idx = torch.arange(kv_len, device=device) // q_len  # 0..T-1
                        T = len(valid_counts)
                        time_decay = decay_rate ** (T - 1 - kv_time_idx)            
                        ms = time_decay.view(B, 1, S).to(dtype=tgt.dtype, device=device) 
                        # ms = torch.ones(B, 1, S, device=device, dtype=tgt.dtype)

                        sim = get_similarity(mk=mk, ms=ms, qk=qk, qe=qe, add_batch_dim=False)
                        sim = sim.transpose(1, 2)  # (B, L, N)
                        sim = sim.masked_fill(bias[None], float('-inf'))

                        affinity = torch.softmax(sim, dim=-1) 

                        V = q_bank   # (1, S, C)
                        out = torch.einsum('bln,bnc->blc', affinity, V)

                        pr_tgt = out

                    else:
                        bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()
                        pr_tgt = ori_queries + self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias)
        if 'is_trkquery' in targets[0] and not targets[0]['is_trkquery']:
            _boxes = []
        
               

        if self.training and self.use_dn:
            input_query_tgt, input_query_bbox, dn_attn_mask, dn_meta =\
                            prepare_for_cdn(targets = targets, dn_cfg = self.dn_cfg, 
                                        num_queries = self.num_queries, hidden_dim = self.hidden_dim, dn_enc = self.dn_enc)
            tgt = torch.cat([input_query_tgt, pr_tgt, tgt], dim=1)
            embedweight = torch.cat([input_query_bbox, pr_points, embedweight], dim=1)

            dn_pad = dn_meta['pad_size']
            # pr_pad = prompt_dn_meta['pad_size']
            pr_pad = pr_tgt.shape[1]
            # pr_pad = prompt_pad_size
            num_queries = self.num_queries
            N = dn_pad + pr_pad + num_queries
            device = dn_attn_mask.device

            dn_s, dn_e = 0, dn_pad
            pr_s, pr_e = dn_e, dn_e + pr_pad
            or_s, or_e = pr_e, pr_e + num_queries

            attn_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
            # attn_mask[:dn_pad, dn_pad:N] = True
            # attn_mask[dn_pad:N, :dn_pad] = True

            # DN ↔ PR
            attn_mask[dn_s:dn_e, pr_s:pr_e] = True
            attn_mask[pr_s:pr_e, dn_s:dn_e] = True
            # DN ↔ ORIG
            attn_mask[dn_s:dn_e, or_s:or_e] = True
            attn_mask[or_s:or_e, dn_s:dn_e] = True

            block_pr_orig = True # 2) PR ↔ ORIG Block 
            if pr_pad > 0 and num_queries > 0 and block_pr_orig:
                # attn_mask[pr_s:pr_e, or_s:or_e] = True
                attn_mask[or_s:or_e, pr_s:pr_e] = True

            # Intra attn mask or block self attn
            dn_block = dn_attn_mask[:dn_pad, :dn_pad] if dn_pad > 0 else None
            # pr_block = pr_attn_mask[:pr_pad, :pr_pad] if pr_pad > 0 else None

            if dn_block is not None:
                attn_mask[:dn_pad, :dn_pad] |= dn_block
                attn_mask[:dn_pad, -num_queries:] = False 

            # if pr_block is not None:
            #     attn_mask[dn_pad:dn_pad+pr_pad, dn_pad:dn_pad+pr_pad] |= pr_block
            attn_mask[dn_pad:dn_pad+pr_pad, -num_queries:] = False
        else:
            # pr_pad = prompt_pad_size
            # attn_mask = pr_attn_mask[:pr_pad, :pr_pad] if pr_pad > 0 else None

            # if self.training or (hasattr(self, 'tracker') and len(self.tracker.tracks) != 0):
            # if self.training:
            if len(_boxes) != 0:
                pr_pad = pr_tgt.shape[1]
                tgt = torch.cat([pr_tgt, tgt], dim=1)                       # [B, pr_pad + num_queries, C]
                embedweight = torch.cat([pr_points, embedweight], dim=1)  # [B, pr_pad + num_queries, ...] (ref coords)

                # pr_pad = prompt_dn_meta['pad_size']          
                num_queries = self.num_queries               
                N = pr_pad + num_queries
                device = pr_tgt.device

                pr_s, pr_e = 0, pr_pad
                or_s, or_e = pr_e, pr_e + num_queries

                attn_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
                attn_mask[or_s:or_e, pr_s:pr_e] = True
                attn_mask[pr_s:pr_e, or_s:or_e] = False 
                attn_mask[0:0+pr_pad, -num_queries:] = False
            else:
                num_queries = self.num_queries               
                device = tgt.device

                or_s, or_e = 0, num_queries

                attn_mask = torch.zeros((num_queries, num_queries), dtype=torch.bool, device=device)

            # pr_block = pr_attn_mask[:pr_pad, :pr_pad] if pr_pad > 0 and pr_attn_mask is not None else None
            # if pr_block is not None:
            #     attn_mask[pr_s:pr_e, pr_s:pr_e] |= pr_block
            # attn_mask = None

        tgt_lens = [tgt.shape[1]]*bs

        hs, reference = self.decoder(memory=final_features, memory_lens=token_lens,
                                         tgt=tgt.flatten(0,1), tgt_lens=tgt_lens,
                                         refpoint_embed=embedweight.flatten(0,1),
                                         pos_embed=pos_embeds,
                                         self_attn_mask = attn_mask)
        if not self.memory_update_early:
            if self.training:
                if self.img_seq_idx == 0:
                    # self.query_bank.append(hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries])
                    self.query_bank.append(hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries].clone().detach())
                else:
                    # ori_queries = hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries]
                    # self.query_bank.append(ori_queries.clone())
                    ori_queries = hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries].clone()
                    self.query_bank.append(ori_queries.detach())
                    device = tgt.device
                    q_bank = torch.cat(self.query_bank, dim=1)
                    q_len = ori_queries.shape[1]
                    q_idx  = torch.arange(q_len, device=device).unsqueeze(1)          # (q,1)
                    kv_idx = (torch.arange(q_bank.shape[1], device=device) % q_len).unsqueeze(0)  # (1,kv)
                    kv_len = q_bank.size(1)       
                    bias = torch.where(q_idx == kv_idx, False, True)
                    # bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()

                    if self.memory_update_type == "sim":
                        B, L, C = ori_queries.shape
                        _, S, _ = q_bank.shape
                        mk = q_bank.permute(0, 2, 1).contiguous()
                        qk = ori_queries.permute(0, 2, 1).contiguous()
                        qe = None
                        ms = torch.ones(B, 1, S, device=device, dtype=tgt.dtype)
                        # T = len(valid_counts)
                        decay_rate = 0.8

                        kv_time_idx = torch.arange(kv_len, device=device) // q_len  # 0..T-1
                        time_decay = decay_rate ** (self.img_seq_idx - 1 - kv_time_idx)            

                        ms = time_decay.view(B, 1, S).to(dtype=tgt.dtype, device=device) 
                        # slot_mask = bias[:, 0].any(dim=1)   # (B,S)  True=mask
                        # ms[:, slot_mask] = 0.0
                        sim = get_similarity(mk=mk, ms=ms, qk=qk, qe=qe, add_batch_dim=False)
                        sim = sim.transpose(1, 2)  # (B, L, N)
                        sim = sim.masked_fill(bias[None], float('-inf'))

                        affinity = torch.softmax(sim/10, dim=-1) 

                        V = q_bank   # (1, S, C)
                        out = torch.einsum('bln,bnc->blc', affinity, V)

                        hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = out
                        # hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias)
                        # cos_sim = F.cosine_similarity(ori_queries[0][:, None],   q_bank,   dim=-1 )  # [2,
                    else:
                        bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()
                        hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = ori_queries + self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias)
            else:
                if 'init_boxes' not in targets[0] and hasattr(self, 'tracker') and len(self.tracker.tracks) != 0 and pr_tgt is not None:
                    frame_buckets = defaultdict(list)
                    pr_pad = pr_tgt.shape[1]
                    query_bank = [v['prev_query_list'] for k, v in self.tracker.tracks.items() if v['age'] == 0]
                    # ori_queries = hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries]
                    ori_queries = hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries].clone()
                    device = tgt.device
                    # query_bank = [torch.cat(sublist, dim=0)[None] for sublist in query_bank]

                    for k, v in self.tracker.tracks.items():
                        if v['age'] != 0:
                            continue
                        # print(k)
                        for t, q in enumerate(v['prev_query_list']):
                            if q is None:
                                continue
                            if isinstance(q, (list, tuple)):
                                q = torch.cat(q, dim=0)  # (n_ti, 768)
                            frame_buckets[t].append(q)   # (n_ti, 768)

                    ori_len = len(frame_buckets)
                    # for k, v in self.tracker.tracks.items():
                    #     if v['age'] < self.tracker.max_age:
                    #         # frame_buckets[ori_len].append(ori_queries[0][k])
                    #         frame_buckets[ori_len].append(ori_queries[0, k].clone())
                    
                    q_n = 0
                    for k, v in self.tracker.tracks.items():
                        if v['age'] == 0:
                            # print(k, q_n)
                            frame_buckets[ori_len].append(ori_queries[0, q_n].clone())
                            q_n += 1

                    ordered_ts = sorted(frame_buckets.keys())
                    frames = [torch.stack(frame_buckets[t]) for t in ordered_ts] 

                    device = frames[0].device
                    dtype  = frames[0].dtype
                    max_len = max(f.shape[0] for f in frames)
                    feat_dim = frames[0].shape[1]

                    valid_counts = []
                    padded_frames = []
                    for f in frames:
                        n = f.shape[0]
                        valid_counts.append(n)
                        if n < max_len:
                            pad = torch.zeros(max_len - n, feat_dim, device=device, dtype=dtype)
                            f = torch.cat([f, pad], dim=0)
                        padded_frames.append(f)  # (max_len, 768)

                    q_bank = torch.stack(padded_frames, dim=0)

                    # if self.img_seq_idx == 1:
                    #     query_bank = torch.stack(query_bank, dim=0)[:, None]
                    # else:
                    #     query_bank = torch.stack(query_bank, dim=0)
                    # q_bank = torch.cat(query_bank, dim=)
                    # q_len = ori_queries.shape[1]
                    # q_idx  = torch.arange(q_len, device=device).unsqueeze(1)          # (q,1)
                    # kv_idx = (torch.arange(q_bank.shape[1], device=device) % q_len).unsqueeze(0)  # (1,kv)
                    # kv_len = q_bank.size(1)       
                    # bias = torch.where(q_idx == kv_idx, False, True)

                    # valid_counts = torch.tensor(valid_counts, device=device) 
                    # kv_time_idx = torch.arange(kv_len, device=device) // q_len        # (kv,)
                    # kv_valid = kv_idx.squeeze(0) < valid_counts.to(device)[kv_time_idx]  # (kv,)
                    # bias = bias | (~kv_valid.unsqueeze(0))

                    q_bank = q_bank.reshape(1, -1, 768)
                    q_len = ori_queries.shape[1]
                    kv_len = q_bank.size(1)

                    q_idx  = torch.arange(q_len, device=device).unsqueeze(1)             # (L, 1)
                    kv_idx = (torch.arange(kv_len, device=device) % q_len).unsqueeze(0)  # (1, S)

                    bias = torch.where(q_idx == kv_idx, False, True)                         # (L_flat, kv)

                    valid_counts = torch.as_tensor(valid_counts, device=device)          # (T,)
                    kv_time_idx  = torch.arange(kv_len, device=device) // q_len          # (S,)
                    kv_valid     = (kv_idx.squeeze(0) < valid_counts[kv_time_idx])       # (S,)
                    bias = bias | (~kv_valid.unsqueeze(0))   

                    if self.memory_update_type == "sim":
                        B, L, C = ori_queries.shape
                        _, S, _ = q_bank.shape
                        mk = q_bank.permute(0, 2, 1).contiguous()
                        qk = ori_queries.permute(0, 2, 1).contiguous()
                        qe = None

                        T = len(valid_counts)
                        decay_rate = 0.8

                        kv_time_idx = torch.arange(kv_len, device=device) // q_len  # 0..T-1
                        time_decay = decay_rate ** (T - 1 - kv_time_idx)            

                        ms = time_decay.view(B, 1, S).to(dtype=tgt.dtype, device=device) 
                        # ms = torch.ones(B, 1, S, device=device, dtype=tgt.dtype)
                        # slot_mask = bias[:, 0].any(dim=1)   # (B,S)  True=mask
                        # ms[:, slot_mask] = 0.0
                        sim = get_similarity(mk=mk, ms=ms, qk=qk, qe=qe, add_batch_dim=False)
                        sim = sim.transpose(1, 2)  # (B, L, N)
                        sim = sim.masked_fill(bias[None], float('-inf'))

                        affinity = torch.softmax(sim/10, dim=-1) 

                        V = q_bank   # (1, S, C)
                        out = torch.einsum('bln,bnc->blc', affinity, V)

                        hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = out

                        # if self.img_seq_idx > 18:
                        #     import pdb; pdb.set_trace()

                        # bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()
                        # hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias).clone()

                        # cos_sim = F.cosine_similarity(ori_queries[0][:, None],q_bank,dim=-1)
                        # cos_sim = F.cosine_similarity(aa[0][:, None],q_bank,dim=-1)
                        # aa = self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias).clone()
                        # import pdb; pdb.set_trace()
                    else:
                        bias = bias.expand(1, self.tracking_query_cross_attn.num_heads, q_len, kv_len).contiguous()
                        hs[-1][:, -(pr_pad + self.num_queries):-self.num_queries] = ori_queries + self.tracking_query_cross_attn(q=ori_queries, k=q_bank, v=q_bank, attn_bias=bias)
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        pred_boxes = torch.stack(outputs_coords)

        outputs_poses = []
        outputs_shapes = []
        outputs_confs = []
        outputs_j3ds = []
        outputs_j2ds = []
        outputs_depths = []
        outputs_querys = []
        outputs_ori_querys = []

        # shape of hs: (lvl, bs, num_queries, dim)
        outputs_pose_6d = self.mean_pose.view(1, 1, -1)
        outputs_shape = self.mean_shape.view(1, 1, -1)
        for lvl in range(hs.shape[0]):

            outputs_pose_6d = outputs_pose_6d + self.pose_head[lvl](hs[lvl])
            outputs_shape = outputs_shape + self.shape_head[lvl](hs[lvl])

            if self.training or lvl == hs.shape[0] - 1:
                outputs_pose = rot6d_to_axis_angle(outputs_pose_6d)

                outputs_conf = self.conf_head(hs[lvl]).sigmoid()
                # # select_queries_idx = torch.where(outputs_conf[0] > 0.5)[0]
                # outputs_pose_6d[0][select_queries_idx]
                # import pdb; pdb.set_trace()

                # cam
                cam_xys = self.cam_head(hs[lvl])

                outputs_vert, outputs_j3d, outputs_j2d, depth, transl\
                = self.process_smpl(poses = outputs_pose,
                                    shapes = outputs_shape,
                                    cam_xys = cam_xys,
                                    cam_intrinsics = cam_intrinsics,
                                    detach_j3ds = detach_j3ds)
                
                outputs_querys.append(hs[lvl])
                outputs_poses.append(outputs_pose)
                outputs_shapes.append(outputs_shape)
                outputs_confs.append(outputs_conf)
                # outputs_verts.append(outputs_vert)
                outputs_j3ds.append(outputs_j3d)
                outputs_j2ds.append(outputs_j2d)
                outputs_depths.append(depth)
        
        outputs_ori_querys.append(tgt)
        pred_poses = torch.stack(outputs_poses)
        pred_betas = torch.stack(outputs_shapes)
        pred_confs = torch.stack(outputs_confs)
        pred_verts = outputs_vert
        pred_transl = transl
        pred_intrinsics = cam_intrinsics
        pred_j3ds = torch.stack(outputs_j3ds)
        pred_j2ds = torch.stack(outputs_j2ds)
        pred_depths = torch.stack(outputs_depths)
        pred_querys = torch.stack(outputs_querys)
        pred_ori_querys = torch.stack(outputs_ori_querys)


        if self.training > 0 and self.use_dn:
            pred_poses, pred_betas,\
            pred_boxes, pred_confs,\
            pred_j3ds, pred_j2ds, pred_depths,\
            pred_verts, pred_transl =\
                dn_post_process(pred_poses, pred_betas,
                                pred_boxes, pred_confs,
                                pred_j3ds, pred_j2ds, pred_depths,
                                pred_verts, pred_transl,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        # if self.training > 0:
        #     pred_poses, pred_betas,\
        #     pred_boxes, pred_confs,\
        #     pred_j3ds, pred_j2ds, pred_depths,\
        #     pred_verts, pred_transl =\
        #         dn_post_process(pred_poses, pred_betas,
        #                         pred_boxes, pred_confs,
        #                         pred_j3ds, pred_j2ds, pred_depths,
        #                         pred_verts, pred_transl,
        #                         prompt_dn_meta, self.aux_loss, self._set_aux_loss)

        # import matplotlib.pyplot as plt
        # x_sorted, _ = torch.sort(pred_confs[-1].squeeze(), descending=True)

        # def knee_threshold(scores: torch.Tensor):
        #     s, _ = torch.sort(scores, descending=True)
        #     N = len(s)
        #     x = torch.linspace(0, 1, N, device=s.device)
        #     y = (s - s.min()) / (s.max() - s.min())  # normalize
        #     line = y[0] + (y[-1] - y[0]) * x
        #     diff = line - y                          
        #     i_star = torch.argmax(diff).item()
        #     tau = s[i_star].item()
        #     return tau, i_star

        # tau, i_star = knee_threshold(x_sorted)
        # # if self.tracker.frame_id != 0:
        # #     self.tracker.conf_thresh = tau
        # # else:
        # #     self.tracker.conf_thresh = tau + 0.2

        # plt.figure(figsize=(6,4))
        # plt.plot(range(len(x_sorted.squeeze().cpu().numpy())), x_sorted.squeeze().cpu().numpy(), marker='o', markersize=3, linewidth=1)
        # # plt.axvline(i_star, color='red', linestyle='--', label=f'knee index = {i_star}')
        # # plt.axhline(tau, color='orange', linestyle='--', label=f'tau = {tau:.3f}')
        # plt.savefig(f'linecharts/thresholding_{self.tracker.frame_id}.jpg')

        out = {'pred_poses': pred_poses[-1], 'pred_betas': pred_betas[-1],
                'pred_boxes': pred_boxes[-1], 'pred_confs': pred_confs[-1], 
               'pred_j3ds': pred_j3ds[-1], 'pred_j2ds': pred_j2ds[-1],
               'pred_verts': pred_verts, 'pred_intrinsics': pred_intrinsics, 
               'pred_depths': pred_depths[-1], 'pred_transl': pred_transl,
               'img': samples}
        if self.training:
            out['kept_indices'] = self._kept_indices
        # Posetrack 0.1 and bedlam 0.05
        j2d_boxes = j2ds_to_bboxes_xywh(out['pred_j2ds'][-1], 0.1)[None]
        out['pred_j2d_boxes'] = j2d_boxes
        out['pred_querys'] = pred_querys[-1]
        out['pred_ori_querys'] = pred_ori_querys[-1]
        
        if self.aux_loss and self.training:
            out['aux_outputs'] = self._set_aux_loss(pred_poses, pred_betas,
                                                    pred_boxes, pred_confs,
                                                    pred_j3ds, pred_j2ds, pred_depths)

        if self.use_sat:
            out['enc_outputs'] = scale_map_dict
        
        out['sat'] = sat_dict

        # out['pr_dn_meta'] = prompt_dn_meta

        if self.training > 0 and self.use_dn:
            out['dn_meta'] = dn_meta

        if 'init_boxes' in targets[0]:
            B = out['pred_confs'].shape[0]
            for b in range(B):
                confs = out['pred_confs'][b].squeeze(-1) if out['pred_confs'][b].dim()>1 else out['pred_confs'][b]  # [Q]
                boxes = out['pred_j2d_boxes'][b]
                kp2ds = out['pred_j2ds'][b]
                Q = boxes.size(0)
                qidx = torch.arange(Q, device=boxes.device)

                querys = out['pred_querys']
                ori_querys = out['pred_ori_querys']
                
                gt_num = len(targets[0]['init_boxes'])
                if self.memory_update_early:
                    self.tracker.add_from_gt(boxes[:gt_num], confs[:gt_num], qidx[:gt_num], kps=kp2ds[:,:25][:gt_num], querys=querys[0][:gt_num], ori_querys=ori_querys[0][:gt_num])
                else:
                    self.tracker.add_from_gt(boxes[:gt_num], confs[:gt_num], qidx[:gt_num], kps=kp2ds[:,:25][:gt_num], querys=querys[0][:gt_num], ori_querys=ori_querys[0][:gt_num])
        elif not self.training and hasattr(self, 'is_tracking_eval') and self.is_tracking_eval:
            B = out['pred_confs'].shape[0]
            keep_indices_per_b = []
            keep_track_ids_per_b = []
            keep_origin_qidx_per_b = []

            for b in range(B):
                confs = out['pred_confs'][b].squeeze(-1) if out['pred_confs'][b].dim()>1 else out['pred_confs'][b]  # [Q]
                
                # if hasattr(self.tracker, 'is_add_new') not self.tracker.is_add_new:
                #     boxes = out['pred_boxes'][b] * self.input_size  # [Q,4]
                # else:
                boxes = out['pred_j2d_boxes'][b]
                # boxes = out['pred_boxes'][b]  * self.input_size 
                kp2ds = out['pred_j2ds'][b]
                Q = boxes.size(0)
                qidx = torch.arange(Q, device=boxes.device)
                querys = out['pred_querys']
                ori_querys = out['pred_ori_querys']


                # image_np = samples[0].detach().to("cpu").permute(1, 2, 0).numpy()
                # mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(samples[0].device)
                # std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(samples[0].device)
                # img = samples[0] * std + mean
                # img = torch.clamp(img, 0, 255).to(torch.uint8).detach().to("cpu").permute(1, 2, 0).numpy()
                # image_np = img[:, :, ::-1]  # RGB2BGR
                # shots = self.shot_detector.process_frame(self.frame_count, image_np)

                # self.frame_count += 1
                # is_new_shot = len(shots) > 1 if self.frame_count > 1 else False
                # print(self.frame_count, is_new_shot)

                # if seq_start and b == 0:
                #     mask = confs >= self.conf_thresh
                #     _ = self.tracker.init_frame(
                #         kp2ds[mask], boxes[mask], confs[mask], qidx[mask], nms_iou=0.5
                #     )

                # active_ids, id2qidx, id2origin = self.tracker.update(kp2ds, boxes, confs, qidx)
                if 'is_gt' in targets[0] and not targets[0]['is_gt'] and hasattr(self.tracker, 'is_add_new') and not self.tracker.is_add_new:
                    gt_boxes = box_cxcywh_to_xyxy(targets[0]['boxes'])
                    gt_boxes[:, 0] = gt_boxes[:, 0] * self.input_size
                    gt_boxes[:, 1] = gt_boxes[:, 1] * self.input_size
                    gt_boxes[:, 2] = gt_boxes[:, 2] * self.input_size
                    gt_boxes[:, 3] = gt_boxes[:, 3] * self.input_size
                    if self.memory_update_early:
                        active_ids, id2qidx, id2origin = self.tracker.update(boxes, confs, qidx, kp2ds[:,:25], querys=querys, gt_boxes=gt_boxes, ori_querys=ori_querys)
                    else:
                        active_ids, id2qidx, id2origin = self.tracker.update(boxes, confs, qidx, kp2ds[:,:25], querys=querys, gt_boxes=gt_boxes, ori_querys=ori_querys)
                else:
                    if self.memory_update_early:
                        active_ids, id2qidx, id2origin = self.tracker.update(boxes, confs, qidx, kp2ds[:,:25], querys=querys, ori_querys=ori_querys)
                    else:
                        active_ids, id2qidx, id2origin = self.tracker.update(boxes, confs, qidx, kp2ds[:,:25], querys=querys, ori_querys=ori_querys)

                if len(id2qidx) == 0:
                    keep_indices = torch.empty((0,), dtype=torch.long, device=boxes.device)
                    keep_track_ids = []
                    keep_origin_qidx = torch.empty((0,), dtype=torch.long, device=boxes.device)
                else:
                    ordered_ids = sorted(id2qidx.keys())
                    keep_indices = torch.tensor(
                        [id2qidx[tid] for tid in ordered_ids],
                        dtype=torch.long, device=boxes.device
                    )
                    keep_origin_qidx = torch.tensor(
                        [id2origin[tid] for tid in ordered_ids],
                        dtype=torch.long, device=boxes.device
                    )
                    keep_track_ids = ordered_ids

                keep_indices_per_b.append(keep_indices)
                keep_track_ids_per_b.append(keep_track_ids)
                keep_origin_qidx_per_b.append(keep_origin_qidx)

            def _slice_field(field):
                if field not in out or out[field] is None:
                    return
                x = out[field]
                if not torch.is_tensor(x):
                    return

                if x.dim() >= 2 and x.size(0) == B:
                    sliced_tensors = []
                    for b in range(B):
                        idxs = keep_indices_per_b[b]  # (Kb,)
                        xb = x[b]
                        if idxs.numel() == 0:
                            empty_shape = (0,) + xb.shape[1:]
                            sliced_tensors.append(xb.new_empty(empty_shape))
                        else:
                            sliced_tensors.append(xb.index_select(0, idxs))
                    out[field] = torch.stack(sliced_tensors, dim=0)

                elif x.dim() >= 1 and (B == 1):
                    idxs = keep_indices_per_b[0]
                    if idxs.numel() == 0:
                        empty_shape = (0,) + x.shape[1:]
                        out[field] = x.new_empty(empty_shape)
                    else:
                        out[field] = x.index_select(0, idxs)
                else:
                    return

            for key in [
                'pred_poses','pred_betas','pred_boxes','pred_confs',
                'pred_j3ds','pred_j2ds','pred_depths','pred_verts',
                'pred_transl', 'pred_j2d_boxes'
            ]:
                _slice_field(key)

            out['track_keep_indices'] = keep_indices_per_b                  # list of LongTensor (per B)
            out['track_origin_qidx']  = keep_origin_qidx_per_b              # list of LongTensor (per B)
            out['track_ids']          = keep_track_ids_per_b   
            out['active_ids'] = active_ids
            out['ori_pred_boxes'] = pred_boxes[-1]

        return out
   
    

    @torch.jit.unused
    def _set_aux_loss(self, pred_poses, pred_betas, pred_boxes, 
                        pred_confs, pred_j3ds, 
                        pred_j2ds, pred_depths):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_poses': a, 'pred_betas': b,
                    'pred_boxes': c, 'pred_confs': d, 
                'pred_j3ds': e, 'pred_j2ds': f, 'pred_depths': g}
                    for a, b, c, d, e, f, g in zip(pred_poses[:-1], pred_betas[:-1], 
                    pred_boxes[:-1], pred_confs[:-1], pred_j3ds[:-1], pred_j2ds[:-1], pred_depths[:-1])]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_sat_video_model(args, set_criterion=True):
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    model = VideoModel(
        encoder,
        decoder,
        num_queries=args.num_queries,
        input_size=args.input_size,
        sat_cfg=args.sat_cfg,
        dn_cfg=args.dn_cfg,
        train_pos_embed=getattr(args,'train_pos_embed',True),
        memory_update_type=args.memory_update_type,
        memory_update_early=args.memory_update_early,
    )


    if set_criterion:
        matcher = build_matcher(args)
        weight_dict = args.weight_dict
        losses = args.losses

        if args.dn_cfg['use_dn']:
            dn_weight_dict = {}
            dn_weight_dict.update({f'{k}_dn': v for k, v in weight_dict.items()})
            weight_dict.update(dn_weight_dict)

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{k}.{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        if args.sat_cfg['use_sat']:
            if 'map_confs' not in weight_dict and 'confs' in weight_dict:
                weight_dict.update({'map_confs': weight_dict['confs']})
            # weight_dict.update({'map_scales': })

        criterion = SetCriterion_SATPR_IMG(matcher, weight_dict, losses = losses, j2ds_norm_scale = args.input_size, num_queries = args.num_queries)
        # criterion = SetCriterion(matcher, weight_dict, losses = losses, j2ds_norm_scale = args.input_size)
        return model, criterion
    else:
        return model, None
