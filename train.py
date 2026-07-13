import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SEN12Dataset
from models.utae import UTAE
from utils.loss import get_seg_loss
from utils.metrics import compute_metrics
from utils.logger import CSVLogger


# ===================== 配置区（要改直接改源码） =====================
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
TRAIN_TXT = 'train.txt'
VAL_TXT = 'val.txt'
DATA_DIR = './data'
JSON_PATH = 'norm.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = './checkpoints'
LOG_DIR = './logs'
GRADIENT_CLIP = 1.0
PRINT_INTERVAL = 10   # 每 N 个 batch 打印一次训练 loss
SAVE_INTERVAL = 10    # 每 N 个 epoch 保存一次模型
NUM_WORKERS = 4
# ====================================================================


class SegmentationTrainer:
    """语义分割训练器。

    负责训练/验证循环、loss 计算（通过 criterion）、
    指标记录（通过 CSVLogger）、模型保存。
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        device,
        save_dir,
        log_dir,
        gradient_clip=1.0,
        print_interval=10,
        save_interval=10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion          # 来自 utils/loss.py 的 get_seg_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.gradient_clip = gradient_clip
        self.print_interval = print_interval
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.logger = CSVLogger(log_dir)

        self.best_iou = 0.0
        self.model.to(self.device)

    def _train_one_epoch(self, epoch):
        """训练一个 epoch，返回平均 train_loss。"""
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs} [Train]')
        for batch_idx, (data, mask) in enumerate(pbar):
            data = data.to(self.device)       # (b, t, c, h, w)
            mask = mask.to(self.device)       # (b, 1, h, w)

            self.optimizer.zero_grad()
            output = self.model(data)         # (b, 1, h, w)
            loss = self.criterion(output, mask)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % self.print_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / total_batches
        return avg_loss

    @torch.no_grad()
    def _validate(self):
        """验证，返回 (val_loss, metrics_dict)。"""
        self.model.eval()
        total_loss = 0.0
        total_batches = len(self.val_loader)

        # 累积整个验证集的所有预测和目标（用于全局计算指标）
        all_outputs = []
        all_masks = []

        for data, mask in tqdm(self.val_loader, desc='[Val]'):
            data = data.to(self.device)
            mask = mask.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, mask)
            total_loss += loss.item()

            all_outputs.append(output.cpu())
            all_masks.append(mask.cpu())

        avg_loss = total_loss / total_batches

        # 拼接所有 batch，一次性计算全局指标
        all_outputs = torch.cat(all_outputs, dim=0)  # (N, 1, h, w)
        all_masks = torch.cat(all_masks, dim=0)       # (N, 1, h, w)
        metrics = compute_metrics(all_outputs, all_masks)

        return avg_loss, metrics

    def train(self):
        """完整训练流程。"""
        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss = self._train_one_epoch(epoch)

            # 验证
            val_loss, metrics = self._validate()

            # 调度器 step
            self.scheduler.step()

            # 记录日志
            self.logger.log_epoch(epoch, train_loss, val_loss, metrics)
            print(
                f'Epoch {epoch:3d}/{self.epochs} | '
                f'Train Loss: {train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'IoU: {metrics["iou"]:.4f} | '
                f'F1: {metrics["f1"]:.4f} | '
                f'Precision: {metrics["precision"]:.4f} | '
                f'Recall: {metrics["recall"]:.4f}'
            )

            # 保存最佳模型（以 IoU 为标准）
            if metrics['iou'] > self.best_iou:
                self.best_iou = metrics['iou']
                best_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_path)
                print(f'  → 新最佳 IoU: {self.best_iou:.4f}，已保存 {best_path}')

            # 定期保存
            if epoch % self.save_interval == 0:
                ckpt_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), ckpt_path)

        print(f'\n训练完成。最佳 IoU: {self.best_iou:.4f}')


# ===================== 主程序入口 =====================
def main():
    # DataLoader
    train_dataset = SEN12Dataset(
        txt_path=TRAIN_TXT,
        data_dir=DATA_DIR,
        json_path=JSON_PATH,
        augment=True,
    )
    val_dataset = SEN12Dataset(
        txt_path=VAL_TXT,
        data_dir=DATA_DIR,
        json_path=JSON_PATH,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 模型
    model = UTAE(in_channels=15, num_classes=1)

    # 优化器 & 调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练器
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=get_seg_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
        gradient_clip=GRADIENT_CLIP,
        print_interval=PRINT_INTERVAL,
        save_interval=SAVE_INTERVAL,
    )
    trainer.train()


if __name__ == '__main__':
    main()
