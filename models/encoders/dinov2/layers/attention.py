# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
import math
from torch import Tensor
from torch import nn

from typing import List, Tuple

import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

class MemEffSelfAttention(SelfAttention):
    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, rope=rope)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        #   def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        # qkv = self.qkv(x)
        # attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        # x = self.proj(attn_v)
        # x = self.proj_drop(x)
        # return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.wq = linear_class(dim, dim, bias=qkv_bias)
        self.wk = linear_class(dim, dim, bias=qkv_bias)
        self.wv = linear_class(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, attn_bias) -> Tensor:
        B, N, C = q.shape
        _, N_K, C = k.shape
        # qkv = qkv_layer(x).reshape(B, N, 3, num_heads, C // num_heads)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        D = C // self.num_heads
        scale = 1.0 / (D ** 0.5)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads) * scale
        k = k.reshape(B, N_K, self.num_heads, C // self.num_heads)
        v = v.reshape(B, N_K, self.num_heads, C // self.num_heads) 
        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # (B, H, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1))

        if attn_bias is not None:
            attn = attn + attn_bias.to(dtype=attn.dtype, device=attn.device)

        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # (B, H, N, D)
        x = torch.matmul(attn, v)
        # (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj_drop(self.proj(x))
        x = self.proj(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = 768
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Linear(inner_dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_bias=None) -> Tensor:
        q = self.to_q(q)
        k = self.to_k(k)
        # v = v.to(q.dtype) 
        v = self.to_v(v)

        B, N, C = q.shape
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), [q, k, v])

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, self.inner_dim])
        x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class MemEffCrossAttentionWeight(nn.Module):
    def __init__(self, dim, heads=1, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = 64
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, q, k, v, attn_bias=None) -> Tensor:
        q = self.to_q(q)
        k = self.to_k(k)
        # v = self.to_v(v)

        B, N_Q, C = q.shape  
        B, N_K, _ = k.shape 

        q, k = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), [q, k])

        q = q * self.scale
        
        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        # v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) 

        # attn_weights = self.attend(scores)
        N_topK = 3
        minus_inf = -1e9 

        k_to_find = scores.size(-1) - N_topK 
        threshold, _ = torch.kthvalue(scores, k=k_to_find, dim=-1, keepdim=True)

        mask = scores < threshold
        scores_masked = scores.masked_fill(mask, minus_inf) 
        attn_weights = torch.softmax(scores_masked, dim=-1)

        # row_sums = attn_weights.sum(dim=-1)
        # print(f"가중치 총합의 평균: {row_sums.mean().item():.6f}") 

        # non_zero_counts = (attn_weights > 1e-6).sum(dim=-1)
        # print(f"0이 아닌 가중치 개수의 평균: {non_zero_counts.float().mean().item():.2f}")

        # attn_weights = self.attend(scores)

        return attn_weights