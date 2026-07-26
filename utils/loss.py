import torch
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