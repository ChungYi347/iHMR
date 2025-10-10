# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .twoway_transformer import TwoWayTransformer

class SMPLXDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        smplx_head_depth: int = 3,
        smplx_head_hidden_dim: int = 256,
        inverse_depth: bool = True,
        use_hands: bool = True,
        use_face: bool = True,
        use_expression: bool = True,
        smpl_mean_params: str = "data/body_models/smpl/smpl_mean_params.npz",
        is_init_pose: bool = False,
    ) -> None:
        """
        Predicts SMPLX given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          smplx_head_depth (int): the depth of the MLP used to predict smplx
          smplx_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict smplx
          inverse_depth (bool): whether to use inverse depth
          use_hands (bool): whether to predict hand poses
          use_face (bool): whether to predict face poses
          use_expression (bool): whether to predict facial expressions
        """
        super().__init__()
        self.inverse_depth = inverse_depth
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            # depth=2,
            depth=4,
            embedding_dim=transformer_dim,
            mlp_dim=1024,
            num_heads=8,
            # attention_block="cross-person",
            attention_block="twoway",
        )
        self.use_hands = use_hands
        self.use_face = use_face
        self.use_expression = use_expression

        embed_dim = 1024
        # self.sam2_prompt_adapter = nn.Sequential(
        #     nn.Conv2d(embed_dim//4, embed_dim, 1, bias=False),
        #     LayerNorm2d(embed_dim),
        # )
        self.sam2_prompt_adapter = nn.Linear(256, 1024, bias=True)
        nn.init.orthogonal_(self.sam2_prompt_adapter.weight) 
        nn.init.zeros_(self.sam2_prompt_adapter.bias)       

        # self.sam2_dense_prompt_adapter = nn.Sequential(
        #     # nn.Conv2d(256, embed_dim, 1, bias=False),
        #     nn.Conv2d(256, embed_dim, 1),
        #     nn.GroupNorm(32, embed_dim),  
        # )

        # SMPLX tokens: body, left_hand, right_hand, face, expression
        self.smpl_token = nn.Embedding(2, transformer_dim)  # smplx_loc
        # self.smplx_token = nn.Embedding(3, transformer_dim) 
        
        # Body parameters (SMPL-X: 53 joints total)
        self.pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 24*6, smplx_head_depth)
        # self.global_orient_head = MLP(transformer_dim, smplx_head_hidden_dim, 1*6, smplx_head_depth)
        # # self.body_pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 22*6, smplx_head_depth) # global_orient(1) + body_pose(21)
        # self.body_pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 21*6, smplx_head_depth) # global_orient(1) + body_pose(21)
        # self.lhand_pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 15*6, smplx_head_depth) # left_hand(15)
        # self.rhand_pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 15*6, smplx_head_depth) # right_hand(15)
        # self.jaw_pose_head = MLP(transformer_dim, smplx_head_hidden_dim, 1*6, smplx_head_depth)   # jaw(1)
        self.shape_head = MLP(transformer_dim, smplx_head_hidden_dim, 10, smplx_head_depth)
        # self.transl_head = MLP(transformer_dim, smplx_head_hidden_dim, 2, smplx_head_depth)
        # self.depth_head = MLP(transformer_dim, smplx_head_hidden_dim, 1, smplx_head_depth)
        self.cam_head = MLP(transformer_dim, smplx_head_hidden_dim, 3, smplx_head_depth)
        
        # Note: Individual hand and face heads are not needed since we predict all 53 joints in pose_head
        # The pose_head already includes all components: body, hands, face
        
        # Expression parameters
        if self.use_expression:
            self.expression_head = MLP(transformer_dim, smplx_head_hidden_dim, 10, smplx_head_depth)

        # self.smpl_mean_params = np.load(smpl_mean_params, allow_pickle=True)
        self.initialize()
        self.is_init_pose = is_init_pose

    def forward(
        self,
        cam_int: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        crossperson: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict SMPLX parameters given image and prompt embeddings.
        
        Returns:
          pose: body pose (22*6)
          shape: body shape (10)
          transl: translation (3)
          transl_c: translation in camera space (2)
          depth_c: depth in camera space (1)
          left_hand_pose: left hand pose (15*6) or None
          right_hand_pose: right hand pose (15*6) or None
          jaw_pose: jaw pose (3) or None
          leye_pose: left eye pose (3) or None
          reye_pose: right eye pose (3) or None
          expression: facial expression (10) or None
          features: transformer features
        """

        # Concatenate output tokens
        # output_tokens = self.smplx_token.weight
        output_tokens = self.smpl_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        sparse_prompt_embeddings = self.sam2_prompt_adapter(sparse_prompt_embeddings)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings.unsqueeze(0), tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe.unsqueeze(0), tokens.shape[0], dim=0)

        # if dense_prompt_embeddings is not None:
        #     dense_prompt_embeddings = self.sam2_dense_prompt_adapter(dense_prompt_embeddings)
        #     src = src + dense_prompt_embeddings

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, crossperson=crossperson)
        smpl_token, loc_token = hs[:,:2,:].permute(1,0,2)
        # hs, src = self.transformer(src, pos_src, tokens, crossperson=crossperson)
        # smpl_token, shape_token, loc_token = hs[:, :3, :].permute(1, 0, 2)
        # features = hs[:,:3,:]
        features = hs[:,:2,:]

        # Body predictions (SMPL-X: 53 joints total)
        # pose = self.pose_head(smpl_token) # + self.init_pose

        body_pose_pred = self.pose_head(smpl_token)
        # body_pose_pred = self.body_pose_head(smpl_token)
        # lhand_pose_pred = self.lhand_pose_head(smpl_token)
        # rhand_pose_pred = self.rhand_pose_head(smpl_token)
        # jaw_pose_pred = self.jaw_pose_head(smpl_token)
        # global_orient_pred = self.global_orient_head(smpl_token)

        shape = self.shape_head(smpl_token) 
        # shape = self.shape_head(smpl_token) + self.init_betas
        # shape = self.shape_head(shape_token) + self.init_betas
        # depth_c = self.depth_head(loc_token)  + self.init_depth
        # transl_c = self.transl_head(loc_token) + self.init_transl
        cam_xys = self.cam_head(loc_token) + self.init_cam

        # transl = self.decode_transl(cam_int, transl_c, depth_c)

        # global_orient_pred += self.init_global_orient
        pose = torch.cat([
            # global_orient_pred.reshape(-1, 1, 6),
            # body_pose_pred.reshape(-1, 21, 6),
            body_pose_pred.reshape(-1, 24, 6),
            # lhand_pose_pred.reshape(-1, 15, 6),
            # rhand_pose_pred.reshape(-1, 15, 6),
            # jaw_pose_pred.reshape(-1, 1, 6),
        ], dim=1)
        # if self.is_init_pose:
        #     pose = pose + self.init_pose

        # Extract individual components from the full pose prediction
        pose_reshaped = pose.reshape(-1, 24, 6)  # [B, 53, 6]
        # pose_reshaped = pose.reshape(-1, 53, 6)  # [B, 53, 6]
        
        # Extract individual poses
        # global_orient = pose_reshaped[:, 0:1, :]  # [B, 1, 6]
        # body_pose = pose_reshaped[:, 1:24, :]     # [B, 21, 6]
        # left_hand_pose = pose_reshaped[:, 22:37, :]  # [B, 15, 6]
        # right_hand_pose = pose_reshaped[:, 37:52, :] # [B, 15, 6] 
        # jaw_pose = pose_reshaped[:, 52:53, :]     # [B, 1, 6]

        # Expression predictions
        # expression = None
        # if self.use_expression:
        #     expression = self.expression_head(smpl_token) + self.init_expression

        # return (pose, shape, transl, transl_c, depth_c, 
        #         left_hand_pose, right_hand_pose, jaw_pose,
        #         expression, features)
        # return (pose, shape, transl, transl_c, depth_c, features)
        return (pose, shape, cam_xys, features)
    

    def decode_transl(self, cam_int, transl, depth):
        focal = cam_int.squeeze()[0,0]
        px, py = transl.unbind(-1)
        pz = depth.unbind(-1)[0]

        if self.inverse_depth:
            pz = 1 / (pz + 1e-6)

        tx = px * pz
        ty = py * pz
        tz = pz * focal / 1000
        t_full = torch.stack([tx, ty, tz], dim=-1)

        return t_full
    
    def _init_weights(self):
        """Initialize weights for all heads with better initialization strategy"""
        # heads_to_init = [self.body_pose_head, self.lhand_pose_head, self.rhand_pose_head, self.jaw_pose_head,
        #                   self.shape_head, self.transl_head, self.depth_head, self.global_orient_head]
        # heads_to_init = [self.pose_head, self.lhand_pose_head, self.rhand_pose_head, self.jaw_pose_head,
        #                 self.shape_head, self.transl_head, self.depth_head]
        # if self.use_expression:
        #     heads_to_init.append(self.expression_head)

        # for head in heads_to_init:
        #     for i, layer in enumerate(head.layers):
        #         if isinstance(layer, nn.Linear):
        #             if i == len(head.layers) - 1:  # Final layer
        #                 # Use small but non-zero initialization for pose heads
        #                 # if head in [self.body_pose_head, self.lhand_pose_head, self.rhand_pose_head, 
        #                 #         self.jaw_pose_head, self.global_orient_head]:
        #                 if head in [self.pose_head, self.lhand_pose_head, self.rhand_pose_head, 
        #                         self.jaw_pose_head]:
        #                     # Small random initialization instead of zero
        #                     nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        #                     if layer.bias is not None:
        #                         # Initialize bias to match init_pose values
        #                         # if head == self.global_orient_head:
        #                         #     nn.init.constant_(layer.bias, 0)  # Will be added to init_pose
        #                         # else:
        #                         #     nn.init.constant_(layer.bias, 0)
        #                         nn.init.constant_(layer.bias, 0)
        #                 else: 
        #                     # Other heads use Xavier with small gain
        #                     nn.init.xavier_uniform_(layer.weight, gain=0.01)
        #                     if layer.bias is not None:
        #                         nn.init.constant_(layer.bias, 0)
        #             else:  # Intermediate layers
        #                 nn.init.xavier_uniform_(layer.weight)
        #                 if layer.bias is not None:
        #                     nn.init.constant_(layer.bias, 0)
        nn.init.xavier_uniform_(self.pose_head.layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.shape_head.layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.cam_head.layers[-1].weight, gain=0.01)
        # nn.init.xavier_uniform_(self.transl_head.layers[-1].weight, gain=0.01)
        # nn.init.xavier_uniform_(self.depth_head.layers[-1].weight, gain=0.01)

        nn.init.constant_(self.pose_head.layers[-1].bias, 0)
        nn.init.constant_(self.shape_head.layers[-1].bias, 0)
        nn.init.constant_(self.cam_head.layers[-1].bias, 0)
        # nn.init.constant_(self.transl_head.layers[-1].bias, 0)
        # nn.init.constant_(self.depth_head.layers[-1].bias, 0)

    def initialize(self):
        """Better initialization with proper pose values"""
        # Improved depth initialization
        if self.inverse_depth:
            init_depth = torch.tensor([[0.1]])  # 1/10 -> more reasonable initial depth
        else:
            init_depth = torch.tensor([[10.]])

        init_transl = torch.tensor([[0.0, 0.0]])
        init_cam = torch.tensor([[0.0, 0.0, 0.1]])
        init_betas = torch.zeros([1, 10])
        # init_betas = torch.tensor([[ 0.20560974,  0.33556297, -0.35068282,  0.35612896,  0.41754073,
        #                             0.03088791,  0.30475676,  0.23613405,  0.20912662,  0.31212646]])
        # init_pose = torch.from_numpy(self.smpl_mean_params['pose'][:]).unsqueeze(0)
        # init_shape = torch.from_numpy(
        #     self.smpl_mean_params['shape'][:].astype('float32')).unsqueeze(0)
        # init_cam = torch.from_numpy(self.smpl_mean_params['cam']).unsqueeze(0)
        
        # Better pose initialization - closer to neutral pose
        init_global_orient = torch.tensor([[1., 0, 0, 0, -1, 0]])  # Identity rotation in 6D
        # init_global_orient = torch.tensor([[1., 0, 0, -1, 0, 0]])  # Identity rotation in 6D
        init_body_pose = torch.tensor([[1., 0, 0, 0, 1, 0]]).repeat(21, 1)  # Neutral body
        
        # More relaxed hand poses (slightly curved)
        # init_lhand_pose = torch.tensor([[1., 0, 0, 0, 1, 0]]).repeat(15, 1) 
        # init_rhand_pose = torch.tensor([[1., 0, 0, 0, 1, 0]]).repeat(15, 1)
        
        # Neutral jaw
        # init_jaw_pose = torch.tensor([[1., 0, 0, 0, 1, 0]])

        # init_body_pose = torch.tensor([[0., 0, 0, 0, 0, 0]]).repeat(52, 1)  # Neutral body
        
        # Combine all poses
        init_pose = torch.cat([
            init_global_orient,
            init_body_pose,
            # init_lhand_pose,
            # init_rhand_pose, 
            # init_jaw_pose
        ], dim=0).unsqueeze(0)  # [1, 53, 6]

        # init_pose = torch.cat([
        #     init_global_orient,
        #     init_body_pose,
        # ], dim=0).unsqueeze(0)  

        # self.register_buffer('init_global_orient', init_global_orient)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)
        # self.register_buffer('init_transl', init_transl)
        # self.register_buffer('init_depth', init_depth)

        if self.use_expression:
            init_expression = torch.zeros([1, 10])
            self.register_buffer('init_expression', init_expression)

        # Initialize weights
        self._init_weights()


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
