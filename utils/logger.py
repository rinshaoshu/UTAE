import os
import csv


class CSVLogger:
    """记录训练过程中的指标到 CSV 文件。

    CSV 列: epoch, train_loss, val_loss, iou, f1, precision, recall
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir: str, 日志保存目录
        """
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'UTAE-metrics.csv')

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'iou', 'f1', 'precision', 'recall'])

    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """
        追加一个 epoch 的记录。

        Args:
            epoch: int, 当前 epoch (从 1 开始)
            train_loss: float, 训练平均 loss
            val_loss: float, 验证平均 loss
            metrics: dict, 包含 iou, f1, precision, recall
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(metrics['iou'], 6),
                round(metrics['f1'], 6),
                round(metrics['precision'], 6),
                round(metrics['recall'], 6),
            ])
