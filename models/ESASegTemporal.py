# -*- coding: utf-8 -*-
"""
CMX-B2 + LTAE2d 时序语义分割模型 (自包含, 仅依赖 torch)
- 输入: (B, T, 15, H, W)，内部按通道切分为 4 模态 [10, 1, 2, 2]
- 架构: CMX-B2 backbone (RGB + 辅助分支均使用 Transformer Block) + LTAE2d 时序融合 + SegFormerHead
- CMX vs CMNeXt 关键区别: 辅助分支使用 Transformer Block 而非 MSPABlock
- 可配置参数: img_size, drop_path_rate
- 仅依赖 torch，完全自包含
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import math


# ===========================================================================
# DropPath
# ===========================================================================

class DropPath(nn.Module):
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(kp) * random_tensor


# ===========================================================================
# ConvLayerNorm
# ===========================================================================

class ConvLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ===========================================================================
# PatchEmbed (RGB + 辅助分支共用，输出 NLC)
# ===========================================================================

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ===========================================================================
# Transformer Block (RGB + 辅助分支共用)
# ===========================================================================

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x
        k, v = self.kv(x_).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DWConv3x3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class TransformerMLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv3x3(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TransformerMLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# ===========================================================================
# PredictorLG — CMX 硬选择 Token Select (辅助模态间逐 token 选择)
# ===========================================================================

class PredictorLG(nn.Module):
    """CMX 风格的 token 选择器: 对每个辅助模态打分, argmax 选择最优模态的 token
    输入: list of (B, N, C) NLC tensors (每个辅助模态一个)
    输出: list of (B, N, 1) scores
    """
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Softmax(dim=-1)
        ) for _ in range(num_modals)])

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        return [self.score_nets[i](x[i]) for i in range(self.num_modals)]


# ===========================================================================
# FRM
# ===========================================================================

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4 // reduction, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max_ = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max_), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        return y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(x.shape[0], 2, 1, x.shape[2], x.shape[3]).permute(1, 0, 2, 3, 4)


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2


# ===========================================================================
# FFM = CrossPath + ChannelEmbed
# ===========================================================================

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        return self.norm1(x1 + self.end_proj1(y1)), self.norm2(x2 + self.end_proj2(y2))


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        return self.channel_emb(merge, H, W)


# ===========================================================================
# CMX Backbone
# ===========================================================================

class CMXBackbone(nn.Module):
    """CMX-B2: RGB + 辅助分支均使用 Transformer Block (NLC 格式)
    embed_dims=[64,128,320,512], depths=[3,4,6,3]
    关键区别于 CMNeXt: 辅助分支使用 Block 而非 MSPABlock, 使用 LayerNorm 而非 ConvLayerNorm
    """

    def __init__(self, num_modals=4, in_channels: List[int] = None,
                 drop_path_rate: float = 0.1):
        super().__init__()
        if in_channels is None:
            in_channels = [3, 3, 3, 3]

        self.num_aux_modals = num_modals - 1
        self.drop_path_rate = drop_path_rate
        embed_dims = [64, 128, 320, 512]
        depths = [3, 4, 6, 3]
        self.channels = embed_dims

        # ---- RGB 分支 PatchEmbed (NLC) ----
        self.rgb_stem = PatchEmbed(in_channels[0], embed_dims[0], 7, 4, 3)

        # ---- 辅助分支 PatchEmbed (NLC, 与 RGB 相同) ----
        if self.num_aux_modals > 0:
            self.aux_stems = nn.ModuleList([
                PatchEmbed(in_channels[i + 1], embed_dims[0], 7, 4, 3)
                for i in range(self.num_aux_modals)
            ])
            # 辅助分支内 stage2/3/4 下采样 (NLC)
            self.aux_downs = nn.ModuleList([
                nn.ModuleList([
                    PatchEmbed(embed_dims[j], embed_dims[j + 1], 3, 2, 1)
                    for _ in range(self.num_aux_modals)
                ]) for j in range(3)
            ])

        # ---- Token Selector: PredictorLG (硬选择 argmax) ----
        if self.num_aux_modals > 1:
            self.token_selectors = nn.ModuleList([
                PredictorLG(embed_dims[i], self.num_aux_modals) for i in range(4)
            ])

        # ---- RGB 分支 Transformer Blocks ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.rgb_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[i]) for i in range(depths[0])])
        self.rgb_norm1 = nn.LayerNorm(embed_dims[0])
        self.rgb_down2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 1)
        self.rgb_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[sum(depths[:1]) + i]) for i in range(depths[1])])
        self.rgb_norm2 = nn.LayerNorm(embed_dims[1])
        self.rgb_down3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 1)
        self.rgb_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[sum(depths[:2]) + i]) for i in range(depths[2])])
        self.rgb_norm3 = nn.LayerNorm(embed_dims[2])
        self.rgb_down4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 1)
        self.rgb_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[sum(depths[:3]) + i]) for i in range(depths[3])])
        self.rgb_norm4 = nn.LayerNorm(embed_dims[3])

        # ---- 辅助分支 Transformer Blocks (CMX: 使用 Block, LayerNorm) ----
        if self.num_aux_modals > 0:
            self.aux_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[i]) for i in range(depths[0])])
            self.aux_norm1 = nn.LayerNorm(embed_dims[0])
            off1 = sum(depths[:1])
            self.aux_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[off1 + i]) for i in range(depths[1])])
            self.aux_norm2 = nn.LayerNorm(embed_dims[1])
            off2 = sum(depths[:2])
            self.aux_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[off2 + i]) for i in range(depths[2])])
            self.aux_norm3 = nn.LayerNorm(embed_dims[2])
            off3 = sum(depths[:3])
            self.aux_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[off3 + i]) for i in range(depths[3])])
            self.aux_norm4 = nn.LayerNorm(embed_dims[3])

            # FRM + FFM 每 stage 一对
            num_heads = [1, 2, 5, 8]
            self.FRMs = nn.ModuleList([FeatureRectifyModule(embed_dims[i], reduction=1) for i in range(4)])
            self.FFMs = nn.ModuleList([FeatureFusionModule(embed_dims[i], reduction=1, num_heads=num_heads[i]) for i in range(4)])

    def _token_select(self, x_aux_nlc: List[Tensor], module: PredictorLG) -> Tensor:
        """CMX 硬选择: 对 NLC list 逐 token argmax 选择最优模态
        x_aux_nlc: list of (B, N, C)
        返回: (B, N, C)  NLC
        """
        scores = module(x_aux_nlc)                        # list of (B, N, 1)
        x_stack = torch.stack(x_aux_nlc, dim=-1)          # (B, N, C, N_modals)
        B, N, C, N_modals = x_stack.shape
        scores_stacked = torch.stack(scores, dim=-1)      # (B, N, 1, N_modals)
        x_index = torch.argmax(scores_stacked, dim=-1)    # (B, N, 1)
        x_index = x_index.unsqueeze(-1).expand(B, N, C, 1)  # (B, N, C, 1)
        x_select = x_stack.gather(-1, x_index)            # (B, N, C, 1)
        return x_select.squeeze(-1)                       # (B, N, C)

    def forward(self, x_list: List[Tensor]) -> List[Tensor]:
        x_rgb = x_list[0]                              # (B, C0, H, W)
        x_aux = x_list[1:] if self.num_aux_modals > 0 else []
        B = x_rgb.shape[0]
        outs = []

        # ======== Stage 1 ========
        x_rgb, H, W = self.rgb_stem(x_rgb)              # NLC
        for blk in self.rgb_block1:
            x_rgb = blk(x_rgb, H, W)
        x1_rgb = self.rgb_norm1(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # NCHW

        if self.num_aux_modals > 0:
            # 辅助模态 stem → 每个返回 (NLC, H, W)
            aux1 = [self.aux_stems[i](x_aux[i])[0] for i in range(self.num_aux_modals)]
            if self.num_aux_modals > 1:
                aux_f = self._token_select(aux1, self.token_selectors[0])  # NLC
            else:
                aux_f = aux1[0]   # NLC
            for blk in self.aux_block1:
                aux_f = blk(aux_f, H, W)
            x1_aux = self.aux_norm1(aux_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # NCHW
            x1_rgb, x1_aux = self.FRMs[0](x1_rgb, x1_aux)
            fused = self.FFMs[0](x1_rgb, x1_aux)       # (B, 64, H/4, W/4)
            outs.append(fused)
            # 辅助特征传给下一 stage (NCHW)
            if self.num_aux_modals > 1:
                aux_feats = [(a.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x1_aux) for a in aux1]
            else:
                aux_feats = [x1_aux]
        else:
            outs.append(x1_rgb)

        # ======== Stage 2 ========
        x_rgb, H, W = self.rgb_down2(x1_rgb)
        for blk in self.rgb_block2:
            x_rgb = blk(x_rgb, H, W)
        x2_rgb = self.rgb_norm2(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        if self.num_aux_modals > 0:
            aux2 = [self.aux_downs[0][i](e)[0] for i, e in enumerate(aux_feats)]
            if self.num_aux_modals > 1:
                aux_f = self._token_select(aux2, self.token_selectors[1])
            else:
                aux_f = aux2[0]
            for blk in self.aux_block2:
                aux_f = blk(aux_f, H, W)
            x2_aux = self.aux_norm2(aux_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x2_rgb, x2_aux = self.FRMs[1](x2_rgb, x2_aux)
            fused = self.FFMs[1](x2_rgb, x2_aux)
            outs.append(fused)
            if self.num_aux_modals > 1:
                aux_feats = [(a.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x2_aux) for a in aux2]
            else:
                aux_feats = [x2_aux]
        else:
            outs.append(x2_rgb)

        # ======== Stage 3 ========
        x_rgb, H, W = self.rgb_down3(x2_rgb)
        for blk in self.rgb_block3:
            x_rgb = blk(x_rgb, H, W)
        x3_rgb = self.rgb_norm3(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        if self.num_aux_modals > 0:
            aux3 = [self.aux_downs[1][i](e)[0] for i, e in enumerate(aux_feats)]
            if self.num_aux_modals > 1:
                aux_f = self._token_select(aux3, self.token_selectors[2])
            else:
                aux_f = aux3[0]
            for blk in self.aux_block3:
                aux_f = blk(aux_f, H, W)
            x3_aux = self.aux_norm3(aux_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x3_rgb, x3_aux = self.FRMs[2](x3_rgb, x3_aux)
            fused = self.FFMs[2](x3_rgb, x3_aux)
            outs.append(fused)
            if self.num_aux_modals > 1:
                aux_feats = [(a.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x3_aux) for a in aux3]
            else:
                aux_feats = [x3_aux]
        else:
            outs.append(x3_rgb)

        # ======== Stage 4 ========
        x_rgb, H, W = self.rgb_down4(x3_rgb)
        for blk in self.rgb_block4:
            x_rgb = blk(x_rgb, H, W)
        x4_rgb = self.rgb_norm4(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        if self.num_aux_modals > 0:
            aux4 = [self.aux_downs[2][i](e)[0] for i, e in enumerate(aux_feats)]
            if self.num_aux_modals > 1:
                aux_f = self._token_select(aux4, self.token_selectors[3])
            else:
                aux_f = aux4[0]
            for blk in self.aux_block4:
                aux_f = blk(aux_f, H, W)
            x4_aux = self.aux_norm4(aux_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x4_rgb, x4_aux = self.FRMs[3](x4_rgb, x4_aux)
            fused = self.FFMs[3](x4_rgb, x4_aux)
            outs.append(fused)
        else:
            outs.append(x4_rgb)

        return outs


# ===========================================================================
# Temporal 机制模块
# ===========================================================================

class PositionalEncoder(nn.Module):
    """正弦位置编码
    PE(pos, 2i)   = sin(pos / 1000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 1000^(2i/d))
    """
    def __init__(self, d: int, T: int = 1000):
        super().__init__()
        pe = torch.zeros(T, d)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, T, d)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.shape[1]]


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力: softmax(Q·K^T / √d_k) · V，支持 pad_mask"""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, pad_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        d_k = q.shape[-1]
        attn = torch.matmul(q, k) / math.sqrt(d_k)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头时序注意力
    Q: 可学习参数 [n_head, d_k]，所有像素共享
    K: Linear(C → n_head*d_k) 从输入投影
    V: 均分 C 给 n_head
    """
    def __init__(self, d_in: int, n_head: int = 16, d_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.trunc_normal_(self.Q, std=0.02)
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        assert d_in % n_head == 0, f"d_in({d_in}) must be divisible by n_head({n_head})"
        self.d_v = d_in // n_head
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, x: Tensor, pad_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        N, T, d_in = x.shape
        K = self.fc1_k(x).view(N, T, self.n_head, self.d_k).permute(0, 2, 3, 1).contiguous()
        V = x.view(N, T, self.n_head, self.d_v).permute(0, 2, 1, 3).contiguous()
        Q = self.Q.view(1, self.n_head, 1, self.d_k).expand(N, -1, -1, -1).contiguous()
        output, attn = self.attention(Q, K, V, pad_mask=pad_mask)
        return output.squeeze(2), attn.squeeze(2)


class LTAE2d(nn.Module):
    """逐像素时序编码器
    输入: (B, T, C, H, W)
    输出: out (B, d_out, H, W), attn (n_head, B, T, H, W)
    """
    def __init__(self, in_channels: int, d_model: int = 256, d_out: int = 128,
                 n_head: int = 16, mlp_ratio: int = 4, dropout: float = 0.1,
                 add_positional_encoding: bool = True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head

        self.inconv = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.in_norm = nn.LayerNorm(d_model)
        self.add_pos = add_positional_encoding
        if add_positional_encoding:
            self.position_encoder = PositionalEncoder(d_model)

        d_k = d_model // n_head
        self.mha = MultiHeadAttention(d_model, n_head=n_head, d_k=d_k, dropout=dropout)

        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_out),
        )
        self.out_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pad_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        B, T, C, H, W = x.shape
        N = B * H * W

        x_flat = x.permute(0, 3, 4, 2, 1).reshape(N, C, T).contiguous()
        x_flat = self.inconv(x_flat)
        x_flat = x_flat.permute(0, 2, 1).contiguous()

        x_flat = self.in_norm(x_flat)
        if self.add_pos:
            x_flat = self.position_encoder(x_flat)

        if pad_mask is not None:
            pad_mask_flat = pad_mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, H, W)
            pad_mask_flat = pad_mask_flat.permute(0, 2, 3, 1).reshape(N, T).contiguous()
        else:
            pad_mask_flat = None

        mha_out, attn = self.mha(x_flat, pad_mask=pad_mask_flat)
        mha_out = mha_out.reshape(N, self.n_head * self.mha.d_v).contiguous()

        out = self.mlp(mha_out)
        out = self.out_norm(out)
        out = self.dropout(out)

        out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        attn = attn.view(B, H, W, self.n_head, T).permute(3, 0, 4, 1, 2).contiguous()

        return out, attn


# ===========================================================================
# SegFormerHead
# ===========================================================================

class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 512, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"mlp{i+1}", nn.Linear(dim, embed_dim))
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
        )
        self.pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: List[Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = []
        for i, f in enumerate(features):
            p = getattr(self, f'mlp{i+1}')(f.flatten(2).transpose(1, 2))
            p = p.transpose(1, 2).reshape(B, -1, *f.shape[-2:])
            if i > 0:
                p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(p)
        seg = self.fuse(torch.cat(outs[::-1], dim=1))
        return self.pred(self.dropout(seg))


# ===========================================================================
# 完整模型: CMXSeg
# ===========================================================================

class CMXSeg(nn.Module):
    """CMX-B2 + LTAE2d 时序语义分割

    输入: (B, T, 15, H, W)  ← T 帧多模态，内部切分 [10, 1, 2, 2]
    流程:
      1. reshape → (B*T, 15, H, W) → 通道分组 → CMX backbone → [f1,f2,f3,f4]
      2. 各特征 reshape 为 (B, T, C_i, H_i, W_i)
      3. LTAE2d 作用于 f4（最深特征）→ temporal_fused + attn
      4. attn 聚合 f1-f3 → 加权平均融合多帧
      5. [f1', f2', f3', temporal_fused] → SegFormerHead → 输出

    输出: (B, num_classes, H, W)
    """

    def __init__(self, num_classes: int = 25, img_size: int = 128,
                 drop_path_rate: float = 0.1,
                 ltae_d_model: int = 256, ltae_d_out: int = 128,
                 ltae_n_head: int = 16, add_pos_encoding: bool = True):
        """
        Args:
            num_classes: 分割类别数
            img_size: 输入图像尺寸 (正方形)
            drop_path_rate: backbone stochastic depth drop rate
            ltae_d_model: LTAE2d 内部维度
            ltae_d_out: LTAE2d 输出维度
            ltae_n_head: LTAE2d 注意力头数
            add_pos_encoding: 是否添加时序位置编码
        """
        super().__init__()
        self.img_size = img_size
        self.drop_path_rate = drop_path_rate
        in_channels = [10, 1, 2, 2]
        self.backbone = CMXBackbone(num_modals=4, in_channels=in_channels,
                                    drop_path_rate=drop_path_rate)

        # LTAE2d 作用于最深 stage 的特征 (embed_dim=512)
        self.ltae = LTAE2d(
            in_channels=self.backbone.channels[-1],   # 512
            d_model=ltae_d_model,                     # 256
            d_out=ltae_d_out,                         # 128
            n_head=ltae_n_head,                       # 16
            add_positional_encoding=add_pos_encoding,
        )

        # SegFormerHead: dims = [64, 128, 320, ltae_d_out]
        head_dims = self.backbone.channels[:3] + [ltae_d_out]
        self.decode_head = SegFormerHead(head_dims, embed_dim=min(512, ltae_d_model * 2), num_classes=num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, ConvLayerNorm, nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _aggregate_with_attn(self, feat: Tensor, attn: Tensor) -> Tensor:
        """用 LTAE2d 的注意力权重聚合多帧特征

        feat: (B, T, C, H, W)  多帧特征
        attn: (n_head, B, T, H_att, W_att)  LTAE2d 输出的注意力
        返回: (B, C, H, W)  加权聚合后的单帧特征
        """
        _, _, _, H, W = feat.shape

        attn_avg = attn.mean(dim=0)  # (B, T, H_att, W_att)

        if H != attn_avg.shape[-2] or W != attn_avg.shape[-1]:
            attn_avg = F.interpolate(attn_avg, size=(H, W), mode='bilinear', align_corners=False)

        attn_weight = attn_avg.unsqueeze(2)  # (B, T, 1, H, W)
        aggregated = (feat * attn_weight).sum(dim=1)  # (B, C, H, W)
        norm = attn_weight.sum(dim=1).clamp(min=1e-8)  # (B, 1, H, W)
        return aggregated / norm

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, 15, H, W)  内部自动切分 [10, 1, 2, 2]
        返回: (B, num_classes, H, W)
        """
        B, T_val, _, H_orig, W_orig = x.shape

        # 1. flatten 时序维: (B*T, 15, H, W)
        x_flat = x.reshape(B * T_val, 15, H_orig, W_orig)

        # 2. 通道分组 + CMX backbone
        x_list = [x_flat[:, :10], x_flat[:, 10:11], x_flat[:, 11:13], x_flat[:, 13:15]]
        features = self.backbone(x_list)  # [f1, f2, f3, f4]  batch = B*T

        # 3. 各特征 reshape 为 (B, T, C_i, H_i, W_i)
        feats_temporal = []
        for f in features:
            _, C, H_f, W_f = f.shape
            feats_temporal.append(f.view(B, T_val, C, H_f, W_f))

        # 4. LTAE2d 作用于最深特征 f4: (B, T, 512, H4, W4)
        f4_temporal, attn = self.ltae(feats_temporal[-1])

        # 5. 用 attn 聚合 f1-f3 多帧特征
        f1_agg = self._aggregate_with_attn(feats_temporal[0], attn)
        f2_agg = self._aggregate_with_attn(feats_temporal[1], attn)
        f3_agg = self._aggregate_with_attn(feats_temporal[2], attn)

        # 6. 送入 SegFormerHead
        head_input = [f1_agg, f2_agg, f3_agg, f4_temporal]
        out = self.decode_head(head_input)
        out = F.interpolate(out, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        return out


# ===========================================================================
# 测试
# ===========================================================================

if __name__ == '__main__':
    img_size = 128
    drop_path_rate = 0.1
    model = CMXSeg(num_classes=25, img_size=img_size, drop_path_rate=drop_path_rate)
    x = torch.randn(1, 3, 15, img_size, img_size)   # (B, T, 15, H, W)
    with torch.no_grad():
        y = model(x)

    print(f"img_size={img_size}, drop_path_rate={drop_path_rate}")
    print(f"输入: (1, 3, 15, {img_size}, {img_size})")
    print(f"  内部切分: ch[0:10]=rgb(10), ch[10:11]=aux1(1), ch[11:13]=aux2(2), ch[13:15]=aux3(2)")
    print(f"输出: {tuple(y.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total / 1e6:.2f} M")
    print(f"Trainable params: {trainable / 1e6:.2f} M")
