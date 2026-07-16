import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(output, target, smooth=1.0):
    """
    计算 Dice 损失。

    Args:
        output: torch.Tensor, shape (b, 1, h, w), 模型输出的 logits
        target: torch.Tensor, shape (b, 1, h, w), 0/1 标签
        smooth: float, 平滑系数，防止分母为0

    Returns:
        loss: torch.Tensor, 标量
    """
    # 将 logits 转换为概率
    prob = torch.sigmoid(output)

    # 展平
    prob_flat = prob.view(-1)
    target_flat = target.view(-1).float()

    # 计算交集和并集
    intersection = (prob_flat * target_flat).sum()
    union = prob_flat.sum() + target_flat.sum()

    # Dice 系数
    dice = (2. * intersection + smooth) / (union + smooth)

    # Dice 损失 = 1 - Dice
    return 1 - dice


def get_seg_loss(output, target):
    """
    计算二分类分割损失：0.5 × BCE(pos_weight=5) + 0.5 × Dice(smooth=1)

    Args:
        output: torch.Tensor, shape (b, 1, h, w), 模型输出的 logits
        target: torch.Tensor, shape (b, 1, h, w), 0/1 标签

    Returns:
        loss: torch.Tensor, 标量
    """

    device = output.device
    pos_weight = torch.tensor([5.0]).to(device)

    # BCE 损失，正样本权重为 5
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce_criterion(output, target.float())

    # Dice 损失，smooth=1
    dice = dice_loss(output, target, smooth=1.0)

    # 组合损失
    loss = 0.5 * bce_loss + 0.5 * dice

    return loss