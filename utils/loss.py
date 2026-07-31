# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class CE_Dice(nn.Module):
    """二分类交叉熵 + Dice Loss"""

    def __init__(self, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        """
        Args:
            logits: (B, 2, H, W)
            target: (B, 2, H, W) one-hot 格式
        """
        target_indices = torch.argmax(target, dim=1)

        ce_loss = F.cross_entropy(logits, target_indices, ignore_index=self.ignore_index)

        probs = torch.softmax(logits, dim=1)
        pred_fore = probs[:, 1, :, :]
        target_fore = target[:, 1, :, :]

        intersection = (pred_fore * target_fore).sum()
        union = pred_fore.sum() + target_fore.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        return ce_loss + dice_loss


class QMFLoss(nn.Module):
    """
    QMF 损失模块 - 二分类语义分割版本

    包含：
    1. ℒ_fused = BCE/Dice(fused_logits, labels)
    2. ℒ_uni = Σ BCE/Dice(logits_m, labels)
    3. ℒ_reg = 采样正则化（强制权重与历史损失负相关）
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
        """
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)
        pred_foreground = probs[:, 1, :, :]

        valid_mask = (labels != self.ignore_index)
        pred_foreground = pred_foreground[valid_mask]
        labels_foreground = labels[valid_mask].float()

        if labels_foreground.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        intersection = (pred_foreground * labels_foreground).sum()
        union = pred_foreground.sum() + labels_foreground.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

    def bce_loss(self, logits, labels):
        """二分类交叉熵损失（忽略 ignore_index）"""
        B, C, H, W = logits.shape
        pred = logits[:, 1, :, :]
        valid_mask = (labels != self.ignore_index)
        pred = pred[valid_mask]
        labels_valid = labels[valid_mask].float()

        if labels_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        return F.binary_cross_entropy_with_logits(pred, labels_valid)

    def compute_seg_loss(self, logits, labels):
        """计算二分类分割损失（BCE + Dice 可选）"""
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

        # 3. 正则化损失
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

        # 4. 总损失
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
        """更新历史损失和权重的滑动窗口"""
        with torch.no_grad():
            B, M, H, W = losses_pixel.shape
            loss_per_sample = []
            weight_per_sample = []

            for b in range(B):
                for m in range(M):
                    # 只计算有效像素的平均损失
                    valid = losses_pixel[b, m] > 0  # ignore_index 位置为 0
                    if valid.sum() > 0:
                        loss_mean = losses_pixel[b, m][valid].mean().item()
                    else:
                        loss_mean = 0.0
                    loss_per_sample.append(loss_mean)

                    # 权重取空间平均
                    weight_mean = weights[b, m].mean().item()
                    weight_per_sample.append(weight_mean)

            # 按模态分组存储
            for m in range(M):
                self.hist_loss[m].extend(loss_per_sample[m::M])
                self.hist_weights[m].extend(weight_per_sample[m::M])

    def _compute_regularization_loss(self, losses_pixel, weights, sample_pairs=30):
        """计算采样正则化损失"""
        B = losses_pixel.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=losses_pixel.device)

        # 对空间维度求平均
        losses_avg = losses_pixel.mean(dim=[2, 3])  # (B, M)
        weights_avg = weights.mean(dim=[2, 3])  # (B, M)

        reg_loss = 0.0
        num_pairs = 0

        for m in range(self.num_modalities):
            # 获取当前模态的历史数据
            hist_loss_m = torch.tensor(self.hist_loss[m], device=losses_pixel.device)
            hist_weight_m = torch.tensor(self.hist_weights[m], device=losses_pixel.device)

            if len(hist_loss_m) < 2:
                continue

            # 从历史数据中采样样本对
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

                # 比较符号 g
                if w_i > w_j:
                    g = 1.0
                elif w_i == w_j:
                    g = 0.0
                else:
                    g = -1.0

                # 正则化损失项
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