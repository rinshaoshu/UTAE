"""
Deep Evidential Fusion with Uncertainty Quantification and Contextual Discounting
for Multimodal Medical Image Segmentation

Consolidated model file (2D version: [B, C, H, W] input).
Reference: Huang et al., "Deep evidential fusion with uncertainty quantification
and reliability learning for multimodal medical image segmentation", Information Fusion, 2025.
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.parameter import Parameter
from einops import rearrange
from copy import deepcopy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



# =============================================================================
# Section 1: Custom autograd functions
# =============================================================================

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


# =============================================================================
# Section 2: Dempster-Shafer Evidence Mapping Module (Ds1)
# =============================================================================

# 在 DsFunction1 类中，修改 backward 方法

class DsFunction1(torch.autograd.Function):
    """
    Custom autograd function implementing Dempster-Shafer evidence mapping
    via prototype-based mass function computation and Dempster's combination rule.

    2D version: input shape [B, C, H, W] (was 3D [B, C, H, W, D]).
    """
    @staticmethod
    def forward(ctx, input, W, BETA, alpha, gamma, class_dim):
        ctx.class_dim = class_dim
        prototype_dim = 20
        [batch_size, in_channel, height, weight] = input.size()

        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, class_dim, device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))

        d = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)
        s = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)
        expo = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)

        mk = torch.cat((
            torch.zeros(class_dim, batch_size, height, weight, device=input.device),
            torch.ones(1, batch_size, height, weight, device=input.device)
        ), 0)

        for k in range(prototype_dim):
            temp = input.permute(1, 0, 2, 3) - torch.mm(
                W[k, :].unsqueeze(1),
                torch.ones(1, batch_size, device=input.device)
            ).unsqueeze(2).unsqueeze(3)
            d[k, :] = 0.5 * (temp * temp).sum(0)
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k, :] = alphap[k] * expo[k, :]
            m = torch.cat((
                U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s[k, :],
                torch.ones(1, batch_size, height, weight, device=input.device) - s[k, :]
            ), 0)

            t2 = mk[:class_dim, :] * (m[:class_dim, :] + torch.ones(class_dim, 1, height, weight, device=input.device) * m[class_dim, :])
            t3 = m[:class_dim, :] * (torch.ones(class_dim, 1, height, weight, device=input.device) * mk[class_dim, :])
            t4 = mk[class_dim, :] * m[class_dim, :].unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        K = mk.sum(0)
        mk_n = (mk / (torch.ones(class_dim + 1, 1, height, weight, device=input.device) * K)).permute(1, 0, 2, 3)
        ctx.save_for_backward(input, W, BETA, alpha, gamma, mk, d)
        return mk_n

    @staticmethod
    def backward(ctx, grad_output):
        input, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors
        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

        class_dim = ctx.class_dim
        prototype_dim = 20
        [batch_size, in_channel, height, weight] = input.size()
        mu = 0
        iw = 1
        grad_output_ = grad_output[:, :class_dim, :, :] * batch_size * class_dim * height * weight

        K = mk.sum(0).unsqueeze(0)
        K2 = K ** 2
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0).unsqueeze(1)
        U = BETA2 / (beta2 * torch.ones(1, class_dim, device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))
        I = torch.eye(class_dim, device=grad_output.device)

        s = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)
        expo = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)
        mm = torch.cat((
            torch.zeros(class_dim, batch_size, height, weight, device=input.device),
            torch.ones(1, batch_size, height, weight, device=input.device)
        ), 0)

        dEdm = torch.zeros(class_dim + 1, batch_size, height, weight, device=input.device)
        dU = torch.zeros(prototype_dim, class_dim, device=input.device)
        Ds = torch.zeros(prototype_dim, batch_size, height, weight, device=input.device)
        DW = torch.zeros(prototype_dim, in_channel, device=input.device)

        for p in range(class_dim):
            dEdm[p, :] = (grad_output_.permute(1, 0, 2, 3) * (
                I[:, p].unsqueeze(1).unsqueeze(2).unsqueeze(3) * K
                - mk[:class_dim, :]
                - 1 / class_dim * (torch.ones(class_dim, 1, height, weight, device=input.device) * mk[class_dim, :])
            )).sum(0) / K2

        dEdm[class_dim, :] = ((grad_output_.permute(1, 0, 2, 3) * (
            -mk[:class_dim, :] + 1 / class_dim * torch.ones(class_dim, 1, height, weight, device=input.device) * (K - mk[class_dim, :])
        )).sum(0)) / K2

        for k in range(prototype_dim):
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k] = alphap[k] * expo[k, :]
            m = torch.cat((
                U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s[k, :],
                torch.ones(1, batch_size, height, weight, device=input.device) - s[k, :]
            ), 0)
            mm[class_dim, :] = mk[class_dim, :] / m[class_dim, :]
            L = torch.ones(class_dim, 1, height, weight, device=input.device) * mm[class_dim, :]
            mm[:class_dim, :] = (mk[:class_dim, :] - L * m[:class_dim, :]) / (m[:class_dim, :] + torch.ones(class_dim, 1, height, weight, device=input.device) * m[class_dim, :])
            R = mm[:class_dim, :] + L
            A = R * torch.ones(class_dim, 1, height, weight, device=input.device) * s[k, :]
            B = U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * torch.ones(1, batch_size, height, weight, device=input.device) * R - mm[:class_dim, :]
            dU[k, :] = torch.mean((A * dEdm[:class_dim, :]).view(class_dim, -1).permute(1, 0), 0)
            Ds[k, :] = (dEdm[:class_dim, :] * B).sum(0) - (dEdm[class_dim, :] * mm[class_dim, :])

            tt1 = Ds[k, :] * (gamma[k] ** 2 * torch.ones(1, batch_size, height, weight, device=input.device)) * s[k, :]
            tt2 = (torch.ones(batch_size, 1, device=input.device) * W[k, :]).unsqueeze(2).unsqueeze(3) - input
            tt1 = tt1.view(1, -1)
            tt2 = tt2.permute(1, 0, 2, 3).reshape(in_channel, batch_size * height * weight).permute(1, 0)
            DW[k, :] = -torch.mm(tt1, tt2)

        DW = iw * DW / (batch_size * height * weight)
        T = beta2 * torch.ones(1, class_dim, device=input.device)
        Dbeta = (2 * BETA / T ** 2) * (dU * (T - BETA2) - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, class_dim, device=input.device) + dU * BETA2)
        Dgamma = -2 * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1) * gamma
        Dalpha = (torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu) * (0.99 * (1 - alphap) * alphap)

        Dinput = torch.zeros(batch_size, in_channel, height, weight, device=input.device)
        temp2 = torch.zeros(prototype_dim, in_channel, height, weight, device=input.device)

        for n in range(batch_size):
            for k in range(prototype_dim):
                test7 = input[n, :] - W[k, :].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                test9 = (Ds[k, n, :, :] * (gamma[k] ** 2) * s[k, n, :, :]).unsqueeze(0).unsqueeze(1)
                temp2[k] = -prototype_dim * test9 * test7
                Dinput[n, :] = temp2.mean(0)

        # ============================================================
        # 修复：必须返回6个梯度，对应forward的6个参数
        # forward: (input, W, BETA, alpha, gamma, class_dim)
        # 其中 class_dim 不需要梯度
        # ============================================================
        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma
        # class_dim 不需要梯度
        grad_class_dim = None

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma, grad_class_dim


class Ds1(nn.Module):
    """
    Dempster-Shafer evidence mapping module.
    Maps feature maps to mass functions using prototypes in feature space.

    Args:
        input_dim: Number of input feature channels
        prototype_dim: Number of prototypes (default 20)
        class_dim: Number of classes (default 4, plus 1 for the empty set)
    """
    def __init__(self, input_dim, prototype_dim, class_dim):
        super(Ds1, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.prototype_dim = prototype_dim
        self.BETA = Parameter(torch.Tensor(self.prototype_dim, self.class_dim))
        self.alpha = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.W = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W)
        nn.init.xavier_uniform_(self.BETA)
        nn.init.constant_(self.gamma, 0.1)
        nn.init.constant_(self.alpha, 0)

    def forward(self, input):
        return DsFunction1.apply(input, self.W, self.BETA, self.alpha, self.gamma, self.class_dim)


# =============================================================================
# Section 3: Swin Transformer Building Blocks (2D)
# =============================================================================

class Mlp(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition 2D features into windows. Input: [B, H, W, C] -> windows: [num_windows, ws, ws, C]."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to 2D features."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Standard Window-based multi-head self-attention with relative position bias (2D)."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table (2D: (2*wh-1) * (2*ww-1) entries)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_embed=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention_kv(nn.Module):
    """Window attention with separate key-value (cross-attention style), 2D."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, skip, x_up, pos_embed=None, mask=None):
        B_, N, C = skip.shape
        kv = self.kv(skip)
        q = x_up

        kv = kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window-based MSA (2D)."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=None)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlock_kv(nn.Module):
    """Swin Transformer block with cross-attention (skip-key-value) for the decoder (2D)."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, skip=None, x_up=None):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.view(B, H, W, C)
        x_up = x_up.view(B, H, W, C)
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b))
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = skip.shape

        if self.shift_size > 0:
            skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            attn_mask = None

        skip = window_partition(skip, self.window_size)
        skip = skip.view(-1, self.window_size * self.window_size, C)
        x_up = window_partition(x_up, self.window_size)
        x_up = x_up.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(skip, x_up, mask=attn_mask, pos_embed=None)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch merging for encoder downsampling (2D)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, 2 * C)
        return x


class Patch_Expanding(nn.Module):
    """Patch expanding for decoder upsampling (2D)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose2d(dim, dim // 2, 2, 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C // 2)
        return x


class project(nn.Module):
    """Projection layer with conv + norm (2D)."""
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding module for 2D images."""
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [patch_size[0] // 2, patch_size[1] // 2]
        stride2 = [patch_size[0] // 2, patch_size[1] // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj1(x)
        x = self.proj2(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Wh, Ww)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer encoder layer (2D)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class BasicLayer_up(nn.Module):
    """A basic Swin Transformer decoder layer (2D, with skip connection and cross-attention)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList()
        self.blocks.append(
            SwinTransformerBlock_kv(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer))
        for i in range(depth - 1):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer))

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)

        return x, H, W


# =============================================================================
# Section 4: Encoder and Decoder (2D)
# =============================================================================

class Encoder(nn.Module):
    """Swin Transformer encoder for 2D medical images."""
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=1, embed_dim=96,
                 depths=[2, 2, 2, 2], num_heads=[4, 8, 16, 32], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size  # (H, W)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        x = self.patch_embed(x)
        down = []
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                down.append(out)
        return down


class Decoder(nn.Module):
    """Swin Transformer decoder for 2D medical images."""
    def __init__(self, pretrain_img_size, embed_dim, patch_size=4, depths=[2, 2, 2],
                 num_heads=[24, 12, 6], window_size=4, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding)
            self.layers.append(layer)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2).contiguous()
            skips[index] = i
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, H, W = layer(x, skips[i], H, W)
            out = x.view(-1, H, W, self.num_features[i])
            outs.append(out)
        return outs


class final_patch_expanding(nn.Module):
    """Final projection layer: expands spatial dimensions and projects to num_classes (2D)."""
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.up(x)
        return x


# ============================================================
# Section 5: Temporal Attention Module (from UTAE)
# ============================================================


class PositionalEncoder(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequences.

    Encodes temporal positions as sin/cos patterns so the model can distinguish
    different time steps in a sequence (e.g. multi-temporal satellite imagery).
    """

    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention for temporal self-attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for temporal sequences.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder (L-TAE) for image time series.

    Attention-based sequence encoding that maps a sequence of images
    to a single feature map. Applied shared across all pixel positions.

    Input:  [B, T, C, H, W]
    Output: [B, d, H, W]
    """

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=None,
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        super(LTAE2d, self).__init__()
        if mlp is None:
            mlp = [256, 128]
        self.in_channels = in_channels
        self.mlp = deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head, num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head, num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp_layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp_layers(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w
        if self.return_att:
            return out, attn
        else:
            return out


class Temporal_Aggregator(nn.Module):
    """Aggregates temporal feature maps into a single map using attention or mean.

    Supports modes: 'att_group', 'att_mean', 'mean'.
    """

    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)
                out = torch.cat([group for group in out], dim=1)
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


# ============================================================
# Section 6: Temporal nnFormer with DS Evidence
# ============================================================


class nnFormerDSTemporal(nn.Module):
    """nnFormer + LTAE2d temporal encoding + DS evidence mapping.

    Input:  [B, T, C, H, W]   multi-temporal satellite images
    Output: [B, num_classes+1, H, W]  mass function (num_classes class masses + 1 ignorance)

    Architecture:
      1. Shared Encoder: each frame [B,C,H,W] → [stage0..stage3] features
      2. LTAE2d at bottleneck: [B,T,bottleneck_dim,h,w] → [B,bottleneck_dim,h,w]
      3. Skip connections: mean aggregation over T per stage
      4. Shared Decoder: neck + aggregated skips → [B,num_classes,H,W]
      5. Ds1 evidence mapping → [B,num_classes+1,H,W] mass function
    """

    def __init__(
        self,

        # === Input / Output ===
        input_channels: int = 15,      # 每帧的输入通道数
        num_classes: int = 2,           # 分割类别数；二分类为2

        # === Image geometry ===
        crop_size: list = [128, 128],  # 输入图像大小 [H, W]

        # === nnFormer backbone ===
        embedding_dim: int = 96,                                 # 基础 embedding 维度
        encoder_depths: list = [2, 2, 2, 2],                    # Encoder 各 stage 的 Swin block 数
        decoder_depths: list = [2, 2, 2],                       # Decoder 各 stage 的 Swin block 数
        encoder_num_heads: list = [3, 6, 12, 24],               # Encoder 各 stage 的 multi-head 数
        patch_size: list = [4, 4],                              # Patch embedding size
        window_size: list = [4, 4, 8, 4],                       # W-MSA 各 stage 的 window size

        # === LTAE temporal attention ===
        n_head: int = 16,                 # Multi-head attention 头数
        d_k: int = 4,                     # 每个 attention head 的 key 维度
        temporal_d_model: int = 256,      # 时序 MLP 中间维度

        # === Training control ===
        freeze_backbone: bool = False,    # 是否冻结 Encoder/Decoder（仅训练 Ds1 + LTAE）
    ):
        super(nnFormerDSTemporal, self).__init__()
        self.num_classes = num_classes

        decoder_num_heads = encoder_num_heads[::-1][:len(decoder_depths)]
        decoder_window_size = window_size[::-1][:len(decoder_depths)]

        # ---- Encoder (shared across time steps) ----
        self.encoder = Encoder(
            pretrain_img_size=crop_size,
            window_size=window_size,
            embed_dim=embedding_dim,
            patch_size=patch_size,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            in_chans=input_channels,
        )

        bottleneck_dim = embedding_dim * (2 ** (len(encoder_depths) - 1))  # e.g. 768

        # ---- LTAE2d: temporal encoding at bottleneck ----
        self.ltae = LTAE2d(
            in_channels=bottleneck_dim,
            n_head=n_head,
            d_k=d_k,
            mlp=[temporal_d_model, bottleneck_dim],
            d_model=temporal_d_model,
            return_att=True,
            positional_encoding=True,
        )

        # ---- Decoder (shared) ----
        self.decoder = Decoder(
            pretrain_img_size=crop_size,
            embed_dim=embedding_dim,
            window_size=decoder_window_size,
            patch_size=patch_size,
            num_heads=decoder_num_heads,
            depths=decoder_depths,
        )

        # ---- Final projection + DS evidence ----
        self.final = nn.ModuleList([
            final_patch_expanding(embedding_dim, num_classes, patch_size=patch_size),
        ])
        self.ds1 = Ds1(input_dim=num_classes, prototype_dim=20, class_dim=num_classes)

        # optional: freeze encoder/decoder weights
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.final.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] multi-temporal input
        Returns:
            mass: [B, num_classes+1, H, W]  DS mass function
                  channels 0..num_classes-1: class evidence masses m_i
                  channel num_classes:        ignorance mass m^Ω
        """
        B, T, C, H, W = x.shape

        # 1. Encode each frame through shared encoder
        x_flat = x.view(B * T, C, H, W)
        skips_flat = self.encoder(x_flat)  # list of [B*T, feat_i, Hi, Wi]

        # Reshape to [B, T, feat, h, w] for temporal processing
        skips = [s.view(B, T, *s.shape[1:]) for s in skips_flat]

        # 2. Temporal encoding at bottleneck
        bottleneck = skips[-1]  # [B, T, 768, H/16, W/16]
        neck, attn = self.ltae(bottleneck)  # [B, 768, H/16, W/16]

        # 3. Aggregate skip connections (mean over T)
        agg_skips = [s.mean(dim=1) for s in skips[:-1]]  # 3 skips: stages 0,1,2

        # 4. Decoder: neck + aggregated skips
        out = self.decoder(neck, agg_skips)
        seg = self.final[0](out[-1])  # [B, num_classes, H, W]
        mass = self.ds1(seg)  # [B, num_classes+1, H, W]  (class masses + ignorance)
        return mass


def temporal_ds(**kwargs):
    """便捷构造函数：nnFormerDSTemporal 二分类。
    
    Usage:
        model = temporal_ds()
        x = torch.randn(2, 10, 15, 128, 128)  # [B, T, C, H, W]
        out = model(x)  # [B, num_classes+1, H, W]  mass function
    """
    return nnFormerDSTemporal(**kwargs)


# ============================================================
# Section 7: 3-Modality Temporal Fusion
# ============================================================


class Fusion3Mod(nn.Module):
    """Contextual discounting + Dempster's rule for 3 modalities (variable class_dim).

    Each modality outputs [B, class_dim+1, H, W] mass function.
    Fused via learned alpha discounts + pointwise product normalization.
    Output: [B, class_dim, H, W]
    """

    def __init__(self, class_dim):
        super(Fusion3Mod, self).__init__()
        self.class_dim = class_dim
        self.alpha1 = Parameter(torch.Tensor(self.class_dim, 1, 1))
        self.alpha2 = Parameter(torch.Tensor(self.class_dim, 1, 1))
        self.alpha3 = Parameter(torch.Tensor(self.class_dim, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha1, 0)
        nn.init.constant_(self.alpha2, 0)
        nn.init.constant_(self.alpha3, 0)

    def forward(self, input1, input2, input3):
        """
        Args:
            input1-3: [B, class_dim+1, H, W] mass functions
        Returns:
            [B, class_dim, H, W] fused class probabilities
        """
        B, C, H, W = input1.shape
        class_dim = self.class_dim
        assert C == class_dim + 1, f"Expected {class_dim+1} channels, got {C}"

        # Extract class masses + spread ignorance mass
        x1 = input1[:, :class_dim] + input1[:, class_dim].unsqueeze(1)
        x2 = input2[:, :class_dim] + input2[:, class_dim].unsqueeze(1)
        x3 = input3[:, :class_dim] + input3[:, class_dim].unsqueeze(1)

        # Sigmoid-bounded discount factors
        alpha1 = 1 / (1 + torch.exp(-self.alpha1))
        alpha2 = 1 / (1 + torch.exp(-self.alpha2))
        alpha3 = 1 / (1 + torch.exp(-self.alpha3))

        batch = torch.ones(B, class_dim, H, W, device=input1.device)
        alpha1 = batch * alpha1
        alpha2 = batch * alpha2
        alpha3 = batch * alpha3

        # Contextual discounting: ax = alpha + (1 - alpha) * x
        ax1 = alpha1 + (1 - alpha1) * x1
        ax2 = alpha2 + (1 - alpha2) * x2
        ax3 = alpha3 + (1 - alpha3) * x3

        # Dempster's rule: pointwise product -> normalize
        pl = ax1 * ax2 * ax3
        K = pl.sum(1, keepdim=True)
        pl = pl / (K + 1e-8)
        return pl


class nnFormerDSTemporalFusion(nn.Module):
    """3-modality temporal fusion model for multi-temporal, multi-spectral segmentation.

    Channel split from 15 input channels:
        Modality 1 (spectral):  channels 0~10            (11 channels)
        Modality 2 (radar #1):  channels [10, 11, 12]   (3 channels, shares ch10)
        Modality 3 (radar #2):  channels [10, 13, 14]   (3 channels, shares ch10)

    Each modality independently → nnFormerDSTemporal → mass [B, num_classes+1, H, W].
    Mass functions fused via Fusion3Mod (contextual discounting + Dempster combination).

    Input:  [B, T, 15, H, W]
    Output (train):     fused, m1_cls, m2_cls, m3_cls    (各 [B, num_classes, H, W])
    Output (inference): fused only                        ([B, num_classes, H, W])
    """

    def __init__(
        self,

        # === Modality split (fixed for SEN12) ===
        ch_mod1: list = None,              # channels for modality 1; default = [0..10]
        ch_mod2: list = None,              # channels for modality 2; default = [10,11,12]
        ch_mod3: list = None,              # channels for modality 3; default = [10,13,14]

        # === Output ===
        num_classes: int = 2,              # 分割类别数

        # === Image geometry ===
        crop_size: list = [128, 128],      # [H, W]

        # === nnFormer backbone (shared across sub-models) ===
        embedding_dim: int = 96,
        encoder_depths: list = [2, 2, 2, 2],
        decoder_depths: list = [2, 2, 2],
        encoder_num_heads: list = [3, 6, 12, 24],
        patch_size: list = [4, 4],
        window_size: list = [4, 4, 8, 4],

        # === LTAE temporal attention (per sub-model) ===
        n_head: int = 8,                   # 时序 attention 头数
        d_k: int = 4,                      # attention key 维度
        temporal_d_model: int = 128,       # 时序 MLP 中间维度

        # === Training control ===
        freeze_backbone: bool = False,      # 是否冻结 Encoder/Decoder
    ):
        super(nnFormerDSTemporalFusion, self).__init__()
        self.num_classes = num_classes

        # resolve channel split
        if ch_mod1 is None:
            ch_mod1 = list(range(0, 11))       # 0~10
        if ch_mod2 is None:
            ch_mod2 = [10, 11, 12]
        if ch_mod3 is None:
            ch_mod3 = [10, 13, 14]
        self.ch_mod1 = ch_mod1
        self.ch_mod2 = ch_mod2
        self.ch_mod3 = ch_mod3

        backbone_kwargs = dict(
            num_classes=num_classes,
            crop_size=crop_size,
            embedding_dim=embedding_dim,
            encoder_depths=encoder_depths,
            decoder_depths=decoder_depths,
            encoder_num_heads=encoder_num_heads,
            patch_size=patch_size,
            window_size=window_size,
            n_head=n_head,
            d_k=d_k,
            temporal_d_model=temporal_d_model,
            freeze_backbone=freeze_backbone,
        )

        # 3 independent temporal DS backbones
        self.mod1 = nnFormerDSTemporal(input_channels=len(ch_mod1), **backbone_kwargs)
        self.mod2 = nnFormerDSTemporal(input_channels=len(ch_mod2), **backbone_kwargs)
        self.mod3 = nnFormerDSTemporal(input_channels=len(ch_mod3), **backbone_kwargs)

        # 3-modality DS fusion
        self.fusion = Fusion3Mod(class_dim=num_classes)

    def forward(self, x, train=True):
        """
        Args:
            x: [B, T, 15, H, W]
            train: bool - 训练时返回所有中间输出用于辅助 loss

        Returns:
            train=True:  (fused, m1_cls, m2_cls, m3_cls)
            train=False: fused only
            各子输出均为 [B, num_classes, H, W] class masses（不含 ignorance）
        """
        x1 = x[:, :, self.ch_mod1, :, :]
        x2 = x[:, :, self.ch_mod2, :, :]
        x3 = x[:, :, self.ch_mod3, :, :]

        # Each sub-model outputs mass [B, num_classes+1, H, W]
        m1 = self.mod1(x1)
        m2 = self.mod2(x2)
        m3 = self.mod3(x3)

        # Fusion: contextual discounting + Dempster combination
        fused = self.fusion(m1, m2, m3)

        if not train:
            return fused
        else:
            nc = self.num_classes
            return fused, m1[:, :nc], m2[:, :nc], m3[:, :nc]



import torch.nn as nn
import torch.nn.functional as F


def dice_loss(output, target_onehot, smooth=1.0):
    """
    计算多类别 Dice 损失（softmax 输出）。

    Args:
        output: torch.Tensor, shape (B, C, H, W), 模型输出的 logits (C >= 2)
        target_onehot: torch.Tensor, shape (B, C, H, W), one-hot 标签
        smooth: float, 平滑系数，防止分母为0

    Returns:
        loss: torch.Tensor, 标量
    """
    prob = F.softmax(output, dim=1)                                  # (B, C, H, W)
    target = target_onehot.float()                                   # (B, C, H, W)

    intersection = (prob * target).sum(dim=(0, 2, 3))                # (C,)
    union = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))    # (C,)

    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_per_class.mean()


class CE_Dice(nn.Module):
    """加权 CrossEntropy + Dice 混合损失（二分类）。

    Loss = 0.5 × CrossEntropy(class_weight=[1,5]) + 0.5 × Dice(smooth=1)

    Input:  output [B, 2, H, W] logits
            target [B, 2, H, W] one-hot
    Output: scalar loss
    """
    def __init__(self):
        super(CE_Dice, self).__init__()
        self.ce_weight = torch.tensor([1.0, 5.0])

    def forward(self, output, target):
        device = output.device
        self.ce_weight = self.ce_weight.to(device)

        ce_loss = F.cross_entropy(
            output, target.argmax(dim=1), weight=self.ce_weight,
        )
        dice = dice_loss(output, target, smooth=1.0)
        return 0.5 * ce_loss + 0.5 * dice


class DSTLoss(nn.Module):
    """Deep Evidential Temporal Fusion 损失。

    Loss = CE_Dice(fused) + CE_Dice(m1) + CE_Dice(m2) + CE_Dice(m3)

    Input:  outputs = (fused, m1, m2, m3)，各 [B, C, H, W] class masses
            target  [B, C, H, W] one-hot
    Output: scalar loss
    """
    def __init__(self):
        super(DSTLoss, self).__init__()
        self.ce_dice = CE_Dice()

    def forward(self, outputs, target):
        fused, m1, m2, m3 = outputs
        return (self.ce_dice(fused, target) +
                self.ce_dice(m1, target) +
                self.ce_dice(m2, target) +
                self.ce_dice(m3, target))


import torch
import torch.nn.functional as F

# ============================================================================
# 1. 创建模型和数据
# ============================================================================

# 创建模型
model = nnFormerDSTemporalFusion(
    num_classes=2,  # 二分类
    crop_size=[128, 128]  # 输入图像尺寸
)

# 创建输入数据
B, T, C, H, W = 2, 15, 15, 128, 128
x = torch.randn(B, T, C, H, W)  # [Batch, Time, Channels, Height, Width]

# 创建标签 (随机生成)
target_indices = torch.randint(0, 2, (B, H, W))  # [B, H, W]
target_onehot = F.one_hot(target_indices, num_classes=2).permute(0, 3, 1, 2).float()  # [B, C, H, W]

print("=" * 60)
print("数据形状检查:")
print(f"输入 x: {x.shape}")  # [2, 15, 15, 128, 128]
print(f"标签 one-hot: {target_onehot.shape}")  # [2, 2, 128, 128]
print("=" * 60)

# ============================================================================
# 2. 模型前向传播 (训练模式)
# ============================================================================

print("\n>>> 模型前向传播 (train=True)")
model.train()
outputs = model(x, train=True)

# outputs 是一个元组: (fused, m1, m2, m3)
fused, m1, m2, m3 = outputs

print(f"fused (融合输出): {fused.shape}")  # [2, 2, 128, 128]
print(f"m1 (模态1输出):   {m1.shape}")  # [2, 2, 128, 128]
print(f"m2 (模态2输出):   {m2.shape}")  # [2, 2, 128, 128]
print(f"m3 (模态3输出):   {m3.shape}")  # [2, 2, 128, 128]

# ============================================================================
# 3. 计算损失
# ============================================================================

print("\n>>> 计算损失")
criterion = DSTLoss()
loss = criterion(outputs, target_onehot)

print(f"总损失: {loss.item():.4f}")

# ============================================================================
# 4. 分解查看损失各分量
# ============================================================================

print("\n>>> 分解损失分量")
ce_dice = CE_Dice()

# 分别计算各模态的损失
loss_fused = ce_dice(fused, target_onehot)
loss_m1 = ce_dice(m1, target_onehot)
loss_m2 = ce_dice(m2, target_onehot)
loss_m3 = ce_dice(m3, target_onehot)

print(f"Loss fused: {loss_fused.item():.6f}")
print(f"Loss m1:    {loss_m1.item():.6f}")
print(f"Loss m2:    {loss_m2.item():.6f}")
print(f"Loss m3:    {loss_m3.item():.6f}")
print(f"总和:       {(loss_fused + loss_m1 + loss_m2 + loss_m3).item():.6f}")
print(f"DSTLoss:    {loss.item():.6f}")

# ============================================================================
# 5. 进一步分解 CE_Dice 损失
# ============================================================================

print("\n>>> 分解 CE_Dice 损失 (以 fused 为例)")
ce_weight = torch.tensor([1.0, 5.0])

# CrossEntropy 部分
ce_loss = F.cross_entropy(fused, target_onehot.argmax(dim=1), weight=ce_weight)


# Dice 部分
def dice_loss(output, target_onehot, smooth=1.0):
    prob = F.softmax(output, dim=1)
    target = target_onehot.float()
    intersection = (prob * target).sum(dim=(0, 2, 3))
    union = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_per_class.mean()


dice = dice_loss(fused, target_onehot, smooth=1.0)

print(f"CE 部分:  {ce_loss.item():.6f}")
print(f"Dice 部分: {dice.item():.6f}")
print(f"CE_Dice:   {0.5 * ce_loss.item() + 0.5 * dice.item():.6f}")
print(f"CE_Dice (直接计算): {loss_fused.item():.6f}")

# ============================================================================
# 6. 完整的训练步
# ============================================================================

print("\n>>> 完整训练步")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):
    optimizer.zero_grad()

    # 前向传播
    outputs = model(x, train=True)

    # 计算损失
    loss = criterion(outputs, target_onehot)

    # 反向传播
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")

# ============================================================================
# 7. 推理模式 (只输出融合结果)
# ============================================================================

print("\n>>> 推理模式 (train=False)")
model.eval()
with torch.no_grad():
    fused_only = model(x, train=False)
    print(f"推理输出 (仅融合): {fused_only.shape}")  # [2, 2, 128, 128]

# ============================================================================
# 8. 可视化输出 (可选)
# ============================================================================

print("\n>>> 输出统计")
print(f"fused: min={fused.min().item():.4f}, max={fused.max().item():.4f}, mean={fused.mean().item():.4f}")
print(f"m1:    min={m1.min().item():.4f}, max={m1.max().item():.4f}, mean={m1.mean().item():.4f}")
print(f"m2:    min={m2.min().item():.4f}, max={m2.max().item():.4f}, mean={m2.mean().item():.4f}")
print(f"m3:    min={m3.min().item():.4f}, max={m3.max().item():.4f}, mean={m3.mean().item():.4f}")

print("\n" + "=" * 60)
print("完成!")