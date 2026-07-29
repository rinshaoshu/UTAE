"""
Taken from https://github.com/roserustowicz/crop-type-mapping/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
    )
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    """3D UNet for semantic segmentation of image time series.

    Input: [B, T, C, H, W]
    Output: [B, num_classes, H, W]

    Example:
        >>> model = UNet3D(
        ...     in_channels=10,
        ...     num_classes=1,
        ...     img_res=128,
        ...     dropout=0.0
        ... )
        >>> x = torch.randn(2, 15, 10, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 1, 128, 128])
    """
    def __init__(self, in_channels, num_classes, img_res=128, dropout=0.0):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_res = img_res
        self.dropout_p = dropout

        feats = 16
        self.en3 = conv_block(self.in_channels, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)

        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)

        self.final = nn.Conv3d(
            feats * 2, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.dropout = nn.Dropout(p=self.dropout_p, inplace=True)

        # Temporal aggregation layer (learnable)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] input tensor

        Returns:
            [B, num_classes, H, W] segmentation map
        """
        # Permute from [B, T, C, H, W] to [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        en3 = self.en3(x)
        pool_3 = self.pool_3(en3)

        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)

        center_in_out = self.center_in(pool_4)
        center_out = self.center_out(center_in_out)

        # Upsample to match en4 dimensions
        center_out = F.interpolate(
            center_out, size=en4.shape[2:], mode="trilinear", align_corners=True
        )
        concat4 = torch.cat([center_out, en4], dim=1)

        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)

        # Upsample to match en3 dimensions
        trans3 = F.interpolate(
            trans3, size=en3.shape[2:], mode="trilinear", align_corners=True
        )
        concat3 = torch.cat([trans3, en3], dim=1)

        dc3 = self.dc3(concat3)
        dc3 = self.dropout(dc3)
        out = self.final(dc3)  # [B, num_classes, T, H, W]

        # Aggregate temporal dimension
        out = self.temporal_pool(out)  # [B, num_classes, 1, H, W]
        out = out.squeeze(2)  # [B, num_classes, H, W]

        # Upsample to original resolution if needed
        if out.shape[-2:] != (self.img_res, self.img_res):
            out = F.interpolate(
                out, size=(self.img_res, self.img_res),
                mode="bilinear", align_corners=True
            )

        return out



class QMF(nn.Module):
    def __init__(self, channel_splits=[11, 3, 3], dim=2):
        """
        Args:
            channel_splits: 三个模型的输入通道数列表
            dim: 通道维度索引 (如果是 BTCHW，dim=2；如果是 BCTHW，dim=1)
        """
        super(QMF, self).__init__()
        self.dim = dim
        self.channel_splits = channel_splits

        self.s2 = UNet3D(in_channels=channel_splits[0], num_classes=2, img_res=128, dropout=0.0)
        self.asc = UNet3D(in_channels=channel_splits[1], num_classes=2, img_res=128, dropout=0.0)
        self.dsc = UNet3D(in_channels=channel_splits[2], num_classes=2, img_res=128, dropout=0.0)

    def QMFfuse(self, s2_logits, asc_logits, dsc_logits):
        """
        输入:
            s2_logits: [B, C, H, W]  - 光谱模态logits
            asc_logits: [B, C, H, W] - 角度模态logits
            dsc_logits: [B, C, H, W] - 距离模态logits
        输出:
            fused_out: [B, C, H, W] - 融合后的输出
            s2_out, asc_out, dsc_out: [B, C, H, W] - 各模态输出
            s2_conf, asc_conf, dsc_conf: [B, 1, H, W] - 各模态置信度
        """
        # 1. 直接使用输入的logits
        s2_out = s2_logits
        asc_out = asc_logits
        dsc_out = dsc_logits

        # 2. 计算各模态置信度（空间自适应）
        s2_conf = -0.1 * torch.logsumexp(s2_out, dim=1, keepdim=True)
        asc_conf = -0.1 * torch.logsumexp(asc_out, dim=1, keepdim=True)
        dsc_conf = -0.1 * torch.logsumexp(dsc_out, dim=1, keepdim=True)

        # 3. 动态融合（加权求和）
        fused_out = s2_out * s2_conf + asc_out * asc_conf + dsc_out * dsc_conf
        return {
            's2': (s2_out, s2_conf),
            'asc': (asc_out, asc_conf),
            'dsc': (dsc_out, dsc_conf),
            'fused': fused_out
        }

    def forward(self, x):
        # 使用 split 自动拆分
        splits = torch.split(x, self.channel_splits, dim=self.dim)

        s2_out = self.s2(splits[0])
        asc_out = self.asc(splits[1])
        dsc_out = self.dsc(splits[2])

        qmf = self.QMFfuse(s2_out, asc_out, dsc_out)
        return qmf