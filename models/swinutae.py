import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy
import numpy as np

# ==================== LTAE2d 模块 ====================
class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])
        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)
        return sinusoid_table

class ScaledDotProductAttention(nn.Module):
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
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn

class LTAE2d(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=4, mlp=[256, 128], dropout=0.2,
                 d_model=256, T=1000, return_att=False, positional_encoding=True):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head
        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.LayerNorm(self.d_model)   # 归一化维度与 d_model 一致
        self.out_norm = nn.LayerNorm(mlp[-1])

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend([nn.Linear(self.mlp[i], self.mlp[i + 1]),
                           nn.BatchNorm1d(self.mlp[i + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).repeat(1, 1, h).unsqueeze(-1).repeat(1, 1, 1, w)
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        # ★ 先投影，再归一化
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)  # [N, T, d_model]

        out = self.in_norm(out)

        if self.positional_encoder is not None and batch_positions is not None:
            pos_enc = self.positional_encoder(batch_positions)
            pos_enc = pos_enc.unsqueeze(1).repeat(1, h*w, 1, 1).view(sz_b * h * w, seq_len, -1)
            out = out + pos_enc

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)
        if self.return_att:
            return out, attn
        else:
            return out

# ==================== Swin Patch Merging（支持时间维度） ====================
class PatchMergingWithTime(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, T, H, W, C = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode='reflect')
            H, W = H + pad_h, W + pad_w

        x = x.reshape(B * T, H, W, C)
        x = x.reshape(B * T, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * T, H//2, W//2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, T, H//2, W//2, -1)
        return x

# ==================== 解码器块（注意力聚合跳跃连接） ====================
class DecoderBlockWithAttn(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, n_head, name=""):
        super().__init__()
        self.name = name
        self.n_head = n_head

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip, attn_weights, target_size=None):
        B, T, C_skip, H_skip, W_skip = skip.shape
        n_head, _, _, H_att, W_att = attn_weights.shape

        # 在 heads 上平均 -> [B, T, H_att, W_att]
        attn_mean = attn_weights.mean(dim=0)

        if target_size is not None:
            H_tgt, W_tgt = target_size
        else:
            H_tgt, W_tgt = H_skip, W_skip

        # ---- 修正点：正确上采样注意力图 ----
        # 将 B 和 T 合并，使插值只作用于空间维度
        B, T, H_att, W_att = attn_mean.shape
        attn_flat = attn_mean.view(B * T, 1, H_att, W_att)   # [B*T, 1, H_att, W_att]
        if H_att != H_tgt or W_att != W_tgt:
            attn_up = F.interpolate(attn_flat, size=(H_tgt, W_tgt), mode='bilinear', align_corners=False)
        else:
            attn_up = attn_flat
        attn_up = attn_up.view(B, T, H_tgt, W_tgt)           # [B, T, H_tgt, W_tgt]
        attn_up = attn_up.unsqueeze(2)                       # [B, T, 1, H_tgt, W_tgt]
        # ---- 修正结束 ----

        # 如果需要，将 skip 调整到目标尺寸
        if H_skip != H_tgt or W_skip != W_tgt:
            skip_resized = F.interpolate(skip.permute(0,1,3,4,2), size=(H_tgt, W_tgt), mode='bilinear', align_corners=False)
            skip_resized = skip_resized.permute(0,1,4,2,3)   # [B, T, C_skip, H_tgt, W_tgt]
        else:
            skip_resized = skip

        # 注意力加权聚合
        skip_agg = (skip_resized * attn_up).sum(dim=1)       # [B, C_skip, H_tgt, W_tgt]

        # 上采样当前特征
        x_up = self.upsample(x)                              # [B, out_channels, H_up, W_up]
        if x_up.shape[2:] != skip_agg.shape[2:]:
            skip_agg = F.interpolate(skip_agg, size=x_up.shape[2:], mode='bilinear', align_corners=False)

        # 拼接并卷积
        out = torch.cat([x_up, skip_agg], dim=1)
        out = self.conv(out)
        return out

# ==================== 修正后的 UNet Head ====================
class SwinUNetHeadWithTemporal(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_classes: int = 1,
        decoder_channels: List[int] = [256, 128, 64],   # 长度 = num_stages-1
        temporal_n_head: int = 16,
        temporal_d_model: int = 256,
        temporal_d_k: int = 4,
        temporal_mlp: List[int] = [256, 128],
        temporal_T: int = 1000,
        debug: bool = False,                     # 新增：控制是否打印调试信息
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.debug = debug                       # 保存

        encoder_dims = [in_channels * (2 ** i) for i in range(self.num_stages)]

        if self.debug:
            print("=== Encoder Stages (Swin Patch Merging) ===")
            for i in range(self.num_stages):
                res = img_size // (2 ** i)
                print(f"  Stage {i}: {res}×{res}, C={encoder_dims[i]} (time kept)")

        self.down_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.down_blocks.append(PatchMergingWithTime(dim=encoder_dims[i]))
            if self.debug:
                print(f"  Down {i}: {encoder_dims[i]} → {encoder_dims[i+1]}")

        self.temporal_encoder = LTAE2d(
            in_channels=encoder_dims[-1],
            n_head=temporal_n_head,
            d_k=temporal_d_k,
            mlp=temporal_mlp,
            d_model=temporal_d_model,
            T=temporal_T,
            return_att=True,
            positional_encoding=True,
        )
        self.temporal_n_head = temporal_n_head
        temporal_out_channels = temporal_mlp[-1]

        if self.debug:
            print(f"\n=== Temporal Encoder (Stage {self.num_stages-1}) ===")
            print(f"  Input: {encoder_dims[-1]}ch, res={img_size//(2**(self.num_stages-1))}")
            print(f"  Output: {temporal_out_channels}ch")

        assert len(decoder_channels) == self.num_stages - 1, \
            f"decoder_channels length {len(decoder_channels)} must equal {self.num_stages-1}"
        self.decoder_channels = decoder_channels

        self.decoder_blocks = nn.ModuleList()
        if self.debug:
            print("\n=== Decoder ===")
        for i in range(self.num_stages - 1):
            in_ch = temporal_out_channels if i == 0 else decoder_channels[i-1]
            skip_idx = self.num_stages - 2 - i
            skip_ch = encoder_dims[skip_idx]
            out_ch = decoder_channels[i]
            if self.debug:
                print(f"  Decoder {i}: {in_ch}→{out_ch} + Skip Stage {skip_idx} (C={skip_ch})")
            self.decoder_blocks.append(
                DecoderBlockWithAttn(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    n_head=temporal_n_head,
                    name=f"Decoder_{i}"
                )
            )

        # 最终融合 + 分割头
        last_decoder_out = decoder_channels[-1]
        self.final_conv = nn.Sequential(
            nn.Conv2d(last_decoder_out + encoder_dims[0], last_decoder_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(last_decoder_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_decoder_out, last_decoder_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(last_decoder_out),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(last_decoder_out, last_decoder_out//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(last_decoder_out//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_decoder_out//2, num_classes, kernel_size=1),
        )
        if self.debug:
            print(f"  Final fusion: {last_decoder_out} + stage0({encoder_dims[0]}) → {last_decoder_out}")

    def forward(self, x, temporal_pad_mask=None):
        if self.debug:
            print("\n" + "="*50)
            print("Forward Pass")
            print("="*50)
            print(f"Input: {x.shape}")

        B, T, C, H, W = x.shape
        batch_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).float()

        x_perm = x.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]
        encoder_features = [x_perm]
        if self.debug:
            print(f"Stage 0: {x_perm.shape}")

        for i, down in enumerate(self.down_blocks):
            x_perm = down(x_perm)
            encoder_features.append(x_perm)
            if self.debug:
                print(f"Stage {i+1}: {x_perm.shape}")

        x_stage3 = encoder_features[-1].permute(0, 1, 4, 2, 3)
        if self.debug:
            print(f"\nStage {self.num_stages-1} before temporal: {x_stage3.shape}")

        out_temporal, attn = self.temporal_encoder(
            x_stage3, batch_positions=batch_positions, pad_mask=temporal_pad_mask
        )
        if self.debug:
            print(f"After temporal: {out_temporal.shape}")
            print(f"Attention: {attn.shape}")

        x_dec = out_temporal
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_idx = self.num_stages - 2 - i
            skip = encoder_features[skip_idx].permute(0, 1, 4, 2, 3)
            target_size = (skip.shape[-2], skip.shape[-1])

            if self.debug:
                print(f"\nDecoder {i}: input {x_dec.shape}, skip {skip.shape}, attn up to {target_size}")
            x_dec = dec_block(x_dec, skip, attn, target_size=target_size)
            if self.debug:
                print(f"  output: {x_dec.shape}")

        if self.debug:
            print(f"\nFinal decoder output: {x_dec.shape}")

        # 聚合 Stage0 跳跃连接
        skip_stage0 = encoder_features[0].mean(dim=1)  # [B,H,W,C0]
        skip_stage0 = skip_stage0.permute(0, 3, 1, 2)  # [B,C0,H,W]
        if self.debug:
            print(f"Skip Stage0 (aggregated): {skip_stage0.shape}")

        # 尺寸检查
        assert x_dec.shape[2:] == skip_stage0.shape[2:], \
            f"Shape mismatch: {x_dec.shape} vs {skip_stage0.shape}"

        x_cat = torch.cat([x_dec, skip_stage0], dim=1)
        x_final = self.final_conv(x_cat)
        seg_logits = self.seg_head(x_final)
        if self.debug:
            print(f"After seg head: {seg_logits.shape} (already at {self.img_size}×{self.img_size})")
            print("="*50 + "\n")
        return seg_logits

# ==================== 测试 ====================
if __name__ == "__main__":
    B, T, C, H, W = 2, 8, 96, 128, 128
    x = torch.randn(B, T, C, H, W)
    model = SwinUNetHeadWithTemporal(
        img_size=128,
        in_channels=96,
        depths=[2, 2, 6, 2],
        num_classes=1,
        decoder_channels=[256, 128, 64],   # 长度 = 3（num_stages-1）
        temporal_n_head=16,
        temporal_d_model=256,
        temporal_d_k=4,
        temporal_mlp=[256, 128],
        temporal_T=1000,
        debug=False
    )
    out = model(x)
    print(f"\nFinal output shape: {out.shape}")  # 应为 [2, 1, 128, 128]