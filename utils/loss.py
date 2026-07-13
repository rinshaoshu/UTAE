import torch.nn as nn


def get_seg_loss(output, target):
    """
    计算二分类分割损失。

    Args:
        output: torch.Tensor, shape (b, 1, h, w), 模型输出的 logits
        target: torch.Tensor, shape (b, 1, h, w), 0/1 标签

    Returns:
        loss: torch.Tensor, 标量
    """
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target.float())
