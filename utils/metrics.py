import torch


def compute_metrics(output, target):
    """
    计算二分类分割指标（micro-average over all pixels）。

    Args:
        output: torch.Tensor, shape (B, 2, H, W), 模型输出的 logits（双通道 softmax）
        target: torch.Tensor, shape (B, 2, H, W), one-hot 标签

    Returns:
        dict: { 'iou': float, 'f1': float, 'precision': float, 'recall': float }
    """
    # softmax argmax 得到类别预测
    pred = torch.argmax(output, dim=1)       # (B, H, W)
    target = target.argmax(dim=1).long()     # (B, 2, H, W) one-hot → (B, H, W) class indices

    # 展平为 1D
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    tp = (pred * target).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()

    eps = 1e-7

    iou = (tp / (tp + fp + fn + eps)).item()
    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    f1 = (2 * precision * recall / (precision + recall + eps))

    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
