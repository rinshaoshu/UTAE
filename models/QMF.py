"""
QMF (Quality-aware Multimodal Fusion) 模块
参考论文: "Provable Dynamic Fusion for Low-Quality Multimodal Data" (ICML 2023)

用于多模态遥感语义分割，支持 3 个模态 (s2, asc, dsc)
输入: (B, T, C, H, W) 其中 T 是时间序列长度
输出: (B, 2, H, W) 二分类分割结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ============================================================
# UNet3D 定义（从 crop-type-mapping 复制）
# ============================================================

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


# ============================================================
# QMF 融合模块
# ============================================================

class QMFFusion(nn.Module):
    """
    QMF 融合模块 - 二分类版本

    输入：3 个模态的 logits 图 (B, 2, H, W)
    输出：融合后的 logits 图 (B, 2, H, W)
    """

    def __init__(self,
                 num_classes=2,
                 num_modalities=3,
                 alpha_init=-0.1,
                 beta_init=0.5,
                 tau=1.0,
                 use_softmax_norm=True):
        super().__init__()

        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.tau = tau
        self.use_softmax_norm = use_softmax_norm

        # ★ 可学习参数：每个模态的 α 和 β (Eq. 9)
        # α < 0（能量越高权重越低），β >= 0（保证权重非负）
        self.alpha = nn.Parameter(torch.ones(num_modalities) * alpha_init)
        self.beta = nn.Parameter(torch.ones(num_modalities) * beta_init)

        # 统计信息（用于监控）
        self.register_buffer('avg_energy', torch.zeros(num_modalities))
        self.register_buffer('avg_weight', torch.zeros(num_modalities))
        self.step_counter = 0

    def compute_energy(self, logits):
        """
        计算能量分数 (Eq. 13)

        二分类: logits (B, 2, H, W)
        energy: (B, H, W)

        能量越低 → 模型越有把握
        """
        log_sum_exp = torch.logsumexp(logits / self.tau, dim=1)
        energy = -self.tau * log_sum_exp
        return energy

    def forward(self, logits_1, logits_2, logits_3):
        """
        三模态融合前向传播

        Args:
            logits_1: (B, 2, H, W) 模态1 的 logits
            logits_2: (B, 2, H, W) 模态2 的 logits
            logits_3: (B, 2, H, W) 模态3 的 logits

        Returns:
            fused_logits: (B, 2, H, W) 融合后的 logits
            weights: (B, 3, H, W) 各模态融合权重
            energies: (B, 3, H, W) 各模态能量分数
        """
        logits_list = [logits_1, logits_2, logits_3]
        B, C, H, W = logits_1.shape
        M = self.num_modalities

        # Step 1: 计算各模态能量分数 (B, H, W)
        energies_list = [self.compute_energy(logits) for logits in logits_list]
        energies = torch.stack(energies_list, dim=1)  # (B, M, H, W)

        # Step 2: 计算融合权重 (Eq. 9)
        # w^m = α^m * u^m + β^m
        raw_weights = self.alpha.view(1, M, 1, 1) * energies + \
                      self.beta.view(1, M, 1, 1)  # (B, M, H, W)

        # ★ 权重归一化
        if self.use_softmax_norm:
            weights = F.softmax(raw_weights, dim=1)  # (B, M, H, W)，和为 1
        else:
            weights = F.relu(raw_weights) + 1e-8
            weights = weights / weights.sum(dim=1, keepdim=True)

        # Step 3: 加权融合
        # fused_logits = Σ w^m * logits^m
        fused_logits = torch.zeros_like(logits_1)
        for m in range(M):
            fused_logits = fused_logits + weights[:, m:m + 1, :, :] * logits_list[m]

        # 更新统计信息
        self._update_stats(energies, weights)

        return fused_logits, weights, energies

    def _update_stats(self, energies, weights):
        """更新统计信息（用于监控）"""
        with torch.no_grad():
            self.step_counter += 1
            momentum = 0.9

            avg_energy = energies.mean(dim=[0, 2, 3])
            avg_weight = weights.mean(dim=[0, 2, 3])

            if self.step_counter == 1:
                self.avg_energy = avg_energy
                self.avg_weight = avg_weight
            else:
                self.avg_energy = momentum * self.avg_energy + (1 - momentum) * avg_energy
                self.avg_weight = momentum * self.avg_weight + (1 - momentum) * avg_weight


# ============================================================
# QMF 主模型
# ============================================================

class QMF(nn.Module):
    """
    QMF 多模态融合模型

    包含三个 UNet3D 编码器 + QMFFusion 融合模块

    输入: (B, T, C_total, H, W)
    输出: (B, 2, H, W) 融合后的 logits

    Args:
        channel_splits: 三个模态的输入通道数列表，如 [11, 3, 3]
        dim: 通道维度索引
             - 如果输入是 (B, T, C, H, W)，dim=2
             - 如果输入是 (B, C, T, H, W)，dim=1
        img_res: 输出图像分辨率
        num_classes: 分类数（默认 2）
        dropout: dropout 率
    """

    def __init__(self,
                 channel_splits=[11, 3, 3],
                 dim=2,
                 img_res=128,
                 num_classes=2,
                 dropout=0.0,
                 alpha_init=-0.1,
                 beta_init=0.5,
                 tau=1.0,
                 use_softmax_norm=True):
        super(QMF, self).__init__()

        self.dim = dim
        self.channel_splits = channel_splits
        self.num_classes = num_classes
        self.img_res = img_res

        # 三个模态的 UNet3D 编码器
        self.s2 = UNet3D(
            in_channels=channel_splits[0],
            num_classes=num_classes,
            img_res=img_res,
            dropout=dropout
        )
        self.asc = UNet3D(
            in_channels=channel_splits[1],
            num_classes=num_classes,
            img_res=img_res,
            dropout=dropout
        )
        self.dsc = UNet3D(
            in_channels=channel_splits[2],
            num_classes=num_classes,
            img_res=img_res,
            dropout=dropout
        )

        # QMF 融合模块
        self.fuse = QMFFusion(
            num_classes=num_classes,
            num_modalities=3,
            alpha_init=alpha_init,
            beta_init=beta_init,
            tau=tau,
            use_softmax_norm=use_softmax_norm
        )

    def forward(self, x):
        """
        Returns:
            fused_logits: (B, 2, H, W) 融合后的 logits
            logits_list: list of 3 tensors, 各模态的 logits
            weights: (B, 3, H, W) 融合权重
            energies: (B, 3, H, W) 能量分数
        """
        splits = torch.split(x, self.channel_splits, dim=self.dim)

        # 各模态编码器前向
        s2_out = self.s2(splits[0])
        asc_out = self.asc(splits[1])
        dsc_out = self.dsc(splits[2])

        logits_list = [s2_out, asc_out, dsc_out]

        # QMF 融合
        fused_logits, weights, energies = self.fuse(s2_out, asc_out, dsc_out)

        # ✅ 返回所有需要的输出
        return fused_logits, logits_list, weights, energies
# ============================================================
# QMF Loss 模块
# ============================================================

class QMFLoss(nn.Module):
    """
    QMF 损失模块 - 二分类语义分割版本

    包含：
    1. ℒ_fused = BCE/Dice(fused_logits, labels)
    2. ℒ_uni = Σ BCE/Dice(logits_m, labels)
    3. ℒ_reg = 采样正则化（强制权重与历史损失负相关）

    支持：
    - 二分类 (num_classes=2)
    - 使用 Dice Loss 处理类别不平衡
    - ignore_index 忽略无效像素
    """

    def __init__(self,
                 num_modalities=3,
                 reg_lambda=0.1,
                 buffer_size=200,
                 use_reg=True,
                 ignore_index=255,
                 use_dice=True,
                 smooth=1e-5):
        super().__init__()

        self.num_modalities = num_modalities
        self.reg_lambda = reg_lambda
        self.use_reg = use_reg
        self.ignore_index = ignore_index
        self.use_dice = use_dice
        self.smooth = smooth

        # 历史损失和权重的滑动窗口
        self.hist_loss = [deque(maxlen=buffer_size) for _ in range(num_modalities)]
        self.hist_weights = [deque(maxlen=buffer_size) for _ in range(num_modalities)]

        # 统计信息
        self.register_buffer('avg_reg_loss', torch.tensor(0.0))
        self.step_counter = 0

    def dice_loss(self, logits, labels):
        """
        二分类 Dice Loss

        Args:
            logits: (B, 2, H, W)
            labels: (B, H, W) 0=背景, 1=前景, ignore_index=255 忽略

        Returns:
            dice_loss: 标量
        """
        B, C, H, W = logits.shape

        # 获取前景类 (class 1) 的概率
        probs = torch.softmax(logits, dim=1)
        pred_foreground = probs[:, 1, :, :]

        # 创建有效掩码（忽略 ignore_index）
        valid_mask = (labels != self.ignore_index)
        pred_foreground = pred_foreground[valid_mask]
        labels_foreground = labels[valid_mask].float()

        if labels_foreground.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Dice 系数
        intersection = (pred_foreground * labels_foreground).sum()
        union = pred_foreground.sum() + labels_foreground.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

    def bce_loss(self, logits, labels):
        """
        二分类交叉熵损失（忽略 ignore_index）
        """
        B, C, H, W = logits.shape
        pred = logits[:, 1, :, :]
        valid_mask = (labels != self.ignore_index)
        pred = pred[valid_mask]
        labels_valid = labels[valid_mask].float()

        if labels_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        return F.binary_cross_entropy_with_logits(pred, labels_valid)

    def compute_seg_loss(self, logits, labels):
        """
        计算二分类分割损失（BCE + Dice 可选）
        """
        ce_loss = self.bce_loss(logits, labels)

        if self.use_dice:
            dice = self.dice_loss(logits, labels)
            return ce_loss + dice
        else:
            return ce_loss

    def forward(self, fused_logits, logits_list, labels, weights):
        """
        计算 QMF 总损失

        Args:
            fused_logits: (B, 2, H, W) 融合后的 logits
            logits_list: list of (B, 2, H, W) 各模态 logits，长度 M
            labels: (B, H, W) 地面真值标签 (0=背景, 1=前景, 255=忽略)
            weights: (B, M, H, W) 融合权重（从 QMFFusion 输出）

        Returns:
            total_loss: 标量
            loss_dict: 各分量损失字典
        """
        M = self.num_modalities
        B, C, H, W = fused_logits.shape

        # 1. 融合损失
        loss_fused = self.compute_seg_loss(fused_logits, labels)

        # 2. 各模态独立损失
        loss_unimodal = 0.0
        losses_per_modality = []
        for m, logits in enumerate(logits_list):
            loss_m = self.compute_seg_loss(logits, labels)
            loss_unimodal = loss_unimodal + loss_m
            losses_per_modality.append(loss_m.detach().item())

        # 3. 正则化损失 (Eq. 16-17)
        loss_reg = torch.tensor(0.0, device=fused_logits.device)

        if self.use_reg and self.training:
            # 计算每个模态每个像素的损失（用于正则化）
            losses_pixel = []
            for logits in logits_list:
                pred = logits[:, 1, :, :]
                loss_m = F.binary_cross_entropy_with_logits(
                    pred, labels.float(), reduction='none'
                )
                loss_m[labels == self.ignore_index] = 0.0
                losses_pixel.append(loss_m)
            losses_pixel = torch.stack(losses_pixel, dim=1)  # (B, M, H, W)

            # 更新历史记录
            self._update_history(losses_pixel, weights)

            # 计算正则化损失
            loss_reg = self._compute_regularization_loss(losses_pixel, weights)

        # 4. 总损失 (Eq. 18)
        total_loss = loss_fused + loss_unimodal + self.reg_lambda * loss_reg

        # 更新统计
        self._update_stats(loss_reg)

        loss_dict = {
            'total': total_loss.item(),
            'fused': loss_fused.item(),
            'unimodal': loss_unimodal.item(),
            'reg': loss_reg.item(),
            'per_modality': losses_per_modality,
        }

        return total_loss, loss_dict

    def _update_history(self, losses_pixel, weights):
        """
        更新历史损失和权重的滑动窗口
        """
        with torch.no_grad():
            B, M, H, W = losses_pixel.shape
            loss_per_sample = []
            weight_per_sample = []

            for b in range(B):
                for m in range(M):
                    # 只计算有效像素的平均损失
                    valid = losses_pixel[b, m] > 0  # ignore_index 位置为 0
                    if valid.sum() > 0:
                        loss_mean = losses_pixel[b, m][valid].mean().item()  # ★ 直接取 item
                    else:
                        loss_mean = 0.0
                    loss_per_sample.append(loss_mean)

                    # 权重取空间平均
                    weight_mean = weights[b, m].mean().item()  # ★ 直接取 item
                    weight_per_sample.append(weight_mean)

            # 按模态分组存储
            for m in range(M):
                self.hist_loss[m].extend(loss_per_sample[m::M])
                self.hist_weights[m].extend(weight_per_sample[m::M])
    def _compute_regularization_loss(self, losses_pixel, weights, sample_pairs=30):
        """
        计算采样正则化损失 (Eq. 16-17)

        核心：κ_i^m >= κ_j^m  =>  w_i^m <= w_j^m
        """
        B = losses_pixel.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=losses_pixel.device)

        # 对空间维度求平均
        losses_avg = losses_pixel.mean(dim=[2, 3])  # (B, M)
        weights_avg = weights.mean(dim=[2, 3])      # (B, M)

        reg_loss = 0.0
        num_pairs = 0

        for m in range(self.num_modalities):
            # 获取当前模态的历史数据
            hist_loss_m = torch.tensor(self.hist_loss[m], device=losses_pixel.device)
            hist_weight_m = torch.tensor(self.hist_weights[m], device=losses_pixel.device)

            if len(hist_loss_m) < 2:
                continue

            # ★ 从历史数据中采样样本对
            hist_len = len(hist_loss_m)
            indices = torch.randint(0, hist_len, (min(sample_pairs * 2, hist_len * 2),))

            for k in range(0, len(indices) - 1, 2):
                i = indices[k]
                j = indices[k + 1]

                if i == j:
                    continue

                kappa_i = hist_loss_m[i]
                kappa_j = hist_loss_m[j]
                w_i = hist_weight_m[i]
                w_j = hist_weight_m[j]

                # Eq. 17: g(wi, wj)
                if w_i > w_j:
                    g = 1.0
                elif w_i == w_j:
                    g = 0.0
                else:
                    g = -1.0

                # Eq. 16: max(0, g*(κi-κj) + |wi-wj|)
                term = g * (kappa_i - kappa_j) + torch.abs(w_i - w_j)
                reg_loss = reg_loss + F.relu(term)
                num_pairs = num_pairs + 1

        if num_pairs > 0:
            reg_loss = reg_loss / num_pairs

        return reg_loss

    def _update_stats(self, loss_reg):
        with torch.no_grad():
            self.step_counter += 1
            momentum = 0.9
            if self.step_counter == 1:
                self.avg_reg_loss = loss_reg.detach()
            else:
                self.avg_reg_loss = momentum * self.avg_reg_loss + (1 - momentum) * loss_reg.detach()

    def reset_history(self):
        """重置历史缓冲区（每个 epoch 开始时调用）"""
        for m in range(self.num_modalities):
            self.hist_loss[m].clear()
            self.hist_weights[m].clear()