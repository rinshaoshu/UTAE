import torch


def compute_metrics(output, target):
    """
    计算二分类分割指标（micro-average over all pixels）。

    Args:
        output: torch.Tensor, shape (b, 1, h, w), 模型输出的 logits
        target: torch.Tensor, shape (b, 1, h, w), 0/1 标签

    Returns:
        dict: { 'iou': float, 'f1': float, 'precision': float, 'recall': float }
    """
    # sigmoid > 0.5 得到二值预测
    pred = (torch.sigmoid(output) > 0.5).long()
    target = target.long()

    # 展平为 1D
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()
    tn = ((1 - pred) * (1 - target)).sum().float()

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
