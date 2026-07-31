import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
import csv
from collections import deque

from dataset.dataset import SEN12Dataset
from models.QMF import QMF, QMFLoss
from utils.loss import CE_Dice

# ===================== 配置区 =====================
BATCH_SIZE = 1
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 100
TRAIN_TXT = 'train.txt'
VAL_TXT = 'val.txt'
DATA_DIR = 'data'
JSON_PATH = 'dataset/norm.json'
BAND = ['s2','asc','dsc']
SAVE_DIR = './checkpoints'
LOG_DIR = './logs'
GRADIENT_CLIP = 1.0
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 20
EARLY_STOP_MONITOR = 'val_f1'
EARLY_STOP_MIN_DELTA = 0.001

# QMF 参数
CHANNEL_SPLITS = [11, 3, 3]  # s2, asc, dsc 的输入通道数
REG_LAMBDA = 0.1
BUFFER_SIZE = 200
USE_REG = True
USE_DICE = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


class CSVLogger:
    """CSV日志记录器"""

    def __init__(self, log_dir, filename='training_log.csv'):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = [
            'epoch',
            'train_loss', 'train_f1', 'train_iou', 'train_precision', 'train_recall',
            'val_loss', 'val_f1', 'val_iou', 'val_precision', 'val_recall',
            'learning_rate', 'best_val_f1',
            # QMF 监控
            'alpha_0', 'alpha_1', 'alpha_2',
            'beta_0', 'beta_1', 'beta_2',
            'weight_0', 'weight_1', 'weight_2',
            'energy_0', 'energy_1', 'energy_2',
            'reg_loss'
        ]

        os.makedirs(log_dir, exist_ok=True)

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_epoch(self, epoch, train_metrics, val_metrics, qmf_stats, lr, best_val_f1):
        row = {
            'epoch': epoch + 1,
            'train_loss': train_metrics.get('train_loss', 0),
            'train_f1': train_metrics.get('train_f1', 0),
            'train_iou': train_metrics.get('train_iou', 0),
            'train_precision': train_metrics.get('train_precision', 0),
            'train_recall': train_metrics.get('train_recall', 0),
            'val_loss': val_metrics.get('val_loss', 0),
            'val_f1': val_metrics.get('val_f1', 0),
            'val_iou': val_metrics.get('val_iou', 0),
            'val_precision': val_metrics.get('val_precision', 0),
            'val_recall': val_metrics.get('val_recall', 0),
            'learning_rate': lr,
            'best_val_f1': best_val_f1,
            'reg_loss': val_metrics.get('val_reg_loss', 0),
        }

        # QMF 统计
        if qmf_stats:
            for i in range(3):
                row[f'alpha_{i}'] = qmf_stats.get(f'alpha_{i}', 0)
                row[f'beta_{i}'] = qmf_stats.get(f'beta_{i}', 0)
                row[f'weight_{i}'] = qmf_stats.get(f'weight_{i}', 0)
                row[f'energy_{i}'] = qmf_stats.get(f'energy_{i}', 0)

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class SegmentationModel:
    def __init__(
            self,
            model,
            loss_fn,
            learning_rate=1e-3,
            weight_decay=0.01,
            gradient_clip=1.0,
            device=DEVICE
    ):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

        self.num_classes = 2

        # 指标
        self.train_metrics = MetricCollection({
            'iou': MulticlassJaccardIndex(num_classes=self.num_classes, average='macro').to(device),
            'f1': MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device),
            'precision': MulticlassPrecision(num_classes=self.num_classes, average='macro').to(device),
            'recall': MulticlassRecall(num_classes=self.num_classes, average='macro').to(device),
        })

        self.val_metrics = MetricCollection({
            'iou': MulticlassJaccardIndex(num_classes=self.num_classes, average='macro').to(device),
            'f1': MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device),
            'precision': MulticlassPrecision(num_classes=self.num_classes, average='macro').to(device),
            'recall': MulticlassRecall(num_classes=self.num_classes, average='macro').to(device),
        })

        self.train_loss = MeanMetric().to(device)
        self.val_loss = MeanMetric().to(device)
        self.val_reg_loss = MeanMetric().to(device)

        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=EPOCHS
        )

    def get_qmf_stats(self):
        """获取 QMF 融合模块的统计信息"""
        stats = {}
        if hasattr(self.model, 'fuse'):
            fuse = self.model.fuse
            alpha = fuse.alpha.detach().cpu().numpy()
            beta = fuse.beta.detach().cpu().numpy()
            weights = fuse.avg_weight.detach().cpu().numpy()
            energies = fuse.avg_energy.detach().cpu().numpy()

            for i in range(3):
                stats[f'alpha_{i}'] = float(alpha[i])
                stats[f'beta_{i}'] = float(beta[i])
                stats[f'weight_{i}'] = float(weights[i])
                stats[f'energy_{i}'] = float(energies[i])

        return stats

    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch索引

        Returns:
            metrics: 包含训练损失和各项指标的字典
        """
        self.model.train()
        self.loss_fn.train()
        self.train_loss.reset()
        self.train_metrics.reset()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]')

        for batch_idx, (data, mask) in enumerate(pbar):
            # 将数据移到设备
            data = data.to(self.device)
            mask = mask.to(self.device)

            # ========== 前向传播 ==========
            # ✅ 一次前向获得所有输出，避免重复计算
            fused_logits, logits_list, weights, energies = self.model(data)
            # fused_logits: (B, 2, H, W) 融合后的logits
            # logits_list: [s2_out, asc_out, dsc_out] 每个都是 (B, 2, H, W)
            # weights: (B, 3, H, W) 融合权重
            # energies: (B, 3, H, W) 能量分数

            # 将mask从one-hot转为索引格式 (B, H, W)
            mask_indices = mask.argmax(dim=1)

            # ========== 计算损失 ==========
            total_loss, loss_dict = self.loss_fn(
                fused_logits,  # 融合后的logits
                logits_list,  # 各模态logits列表
                mask_indices,  # ground truth索引
                weights  # 融合权重（用于正则化）
            )
            # loss_dict 包含: total, fused, unimodal, reg, per_modality

            # ========== 反向传播 ==========
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # ========== 更新指标 ==========
            # 更新损失指标
            self.train_loss.update(total_loss)

            # 更新分割指标 (IoU, F1, Precision, Recall)
            pred_indices = fused_logits.argmax(dim=1)  # (B, H, W)
            self.train_metrics.update(pred_indices, mask_indices)

            # ========== 更新进度条 ==========
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'fused': f'{loss_dict["fused"]:.4f}',
                'uni': f'{loss_dict["unimodal"]:.4f}',
                'reg': f'{loss_dict["reg"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # ========== 可选：定期打印详细信息 ==========
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f'\n  Batch {batch_idx}/{len(train_loader)}')
                print(f'    Total Loss: {total_loss.item():.4f}')
                print(f'    Fused Loss: {loss_dict["fused"]:.4f}')
                print(f'    Unimodal Loss: {loss_dict["unimodal"]:.4f}')
                print(f'    Reg Loss: {loss_dict["reg"]:.4f}')
                print(f'    Per Modality: {[f"{x:.4f}" for x in loss_dict["per_modality"]]}')

                # 打印融合权重统计
                if weights is not None:
                    avg_weights = weights.mean(dim=[0, 2, 3]).detach().cpu().numpy()
                    print(
                        f'    Avg Weights: S2={avg_weights[0]:.3f}, ASC={avg_weights[1]:.3f}, DSC={avg_weights[2]:.3f}')

                # 打印能量统计
                if energies is not None:
                    avg_energies = energies.mean(dim=[0, 2, 3]).detach().cpu().numpy()
                    print(
                        f'    Avg Energies: S2={avg_energies[0]:.3f}, ASC={avg_energies[1]:.3f}, DSC={avg_energies[2]:.3f}')

        # ========== 计算epoch平均指标 ==========
        avg_loss = self.train_loss.compute()
        metrics = self.train_metrics.compute()
        metrics = {f'train_{k}': v.item() for k, v in metrics.items()}
        metrics['train_loss'] = avg_loss.item()

        # 重置指标（为下一个epoch准备）
        self.train_loss.reset()
        self.train_metrics.reset()

        return metrics

    @torch.no_grad()
    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        """
        验证一个epoch - 修复重复前向问题

        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch索引

        Returns:
            metrics: 包含验证损失和各项指标的字典
        """
        self.model.eval()
        self.loss_fn.eval()
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_reg_loss.reset()

        pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Val]')

        for data, mask in pbar:
            data = data.to(self.device)
            mask = mask.to(self.device)

            # ========== ✅ 一次前向获得所有输出 ==========
            # 假设 QMF.forward 返回 (fused_logits, logits_list, weights, energies)
            fused_logits, logits_list, weights, energies = self.model(data)
            # fused_logits: (B, 2, H, W) 融合后的logits
            # logits_list: [s2_out, asc_out, dsc_out] 各模态logits
            # weights: (B, 3, H, W) 融合权重
            # energies: (B, 3, H, W) 能量分数

            # 将mask从one-hot转为索引格式
            mask_indices = mask.argmax(dim=1)  # (B, H, W)

            # ========== 计算损失 ==========
            total_loss, loss_dict = self.loss_fn(
                fused_logits,  # 融合后的logits
                logits_list,  # 各模态logits列表
                mask_indices,  # ground truth索引
                weights  # 融合权重（用于正则化）
            )

            # ========== 更新指标 ==========
            self.val_loss.update(total_loss)
            self.val_reg_loss.update(loss_dict['reg'])

            pred_indices = fused_logits.argmax(dim=1)  # (B, H, W)
            self.val_metrics.update(pred_indices, mask_indices)

            # ========== 更新进度条 ==========
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'reg': f'{loss_dict["reg"]:.4f}'
            })

        # ========== 计算epoch平均指标 ==========
        avg_loss = self.val_loss.compute()
        metrics = self.val_metrics.compute()
        metrics = {f'val_{k}': v.item() for k, v in metrics.items()}
        metrics['val_loss'] = avg_loss.item()
        metrics['val_reg_loss'] = self.val_reg_loss.compute().item()

        # 重置指标
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_reg_loss.reset()

        return metrics

    def save_checkpoint(self, path, epoch, best_f1=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint, path)


def get_data_loaders():
    """创建数据加载器"""
    print("加载训练数据集...")
    train_dataset = SEN12Dataset(
        txt_path=TRAIN_TXT,
        data_dir=DATA_DIR,
        norm_dir=JSON_PATH,
        band=BAND,
        augment=True,
    )
    print(f"训练集大小: {len(train_dataset)}")

    print("加载验证数据集...")
    val_dataset = SEN12Dataset(
        txt_path=VAL_TXT,
        data_dir=DATA_DIR,
        norm_dir=JSON_PATH,
        band=BAND,
        augment=False,
    )
    print(f"验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def train():
    """训练主循环"""
    # ========== 1. 创建模型 ==========
    model = QMF(
        channel_splits=CHANNEL_SPLITS,
        dim=2  # BTCHW 中通道维是 dim=2
    )

    # ========== 2. 创建 QMF Loss ==========
    loss_fn = QMFLoss(
        num_modalities=3,
        reg_lambda=REG_LAMBDA,
        buffer_size=BUFFER_SIZE,
        use_reg=USE_REG,
        ignore_index=255,
        use_dice=USE_DICE,
    )

    # ========== 3. 创建训练器 ==========
    trainer = SegmentationModel(
        model=model,
        loss_fn=loss_fn,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        device=DEVICE
    )

    # ========== 4. 加载数据 ==========
    train_loader, val_loader = get_data_loaders()

    # ========== 5. 创建保存目录 ==========
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_logger = CSVLogger(LOG_DIR, 'training_log.csv')

    # ========== 6. 训练循环 ==========
    best_val_f1 = -float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\n开始训练 QMF，共 {EPOCHS} 个 epoch")
    print(f"  - 模态数: 3 (s2, asc, dsc)")
    print(f"  - 输入通道: {CHANNEL_SPLITS}")
    print(f"  - reg_lambda: {REG_LAMBDA}")
    print(f"  - use_dice: {USE_DICE}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # 验证
        val_metrics = trainer.validate_epoch(val_loader, epoch)

        # 更新学习率
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # 获取 QMF 统计
        qmf_stats = trainer.get_qmf_stats()

        # 检查最佳模型
        if val_metrics['val_f1'] > best_val_f1 + EARLY_STOP_MIN_DELTA:
            best_val_f1 = val_metrics['val_f1']
            best_epoch = epoch
            patience_counter = 0

            checkpoint_path = os.path.join(SAVE_DIR, 'best_model.pth')
            trainer.save_checkpoint(checkpoint_path, epoch, best_val_f1)
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        # 记录日志
        csv_logger.log_epoch(epoch, train_metrics, val_metrics, qmf_stats, current_lr, best_val_f1)

        # 打印
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}, F1: {train_metrics['train_f1']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}, F1: {val_metrics['val_f1']:.4f}, IoU: {val_metrics['val_iou']:.4f}")
        print(f"  Reg Loss: {val_metrics['val_reg_loss']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        if qmf_stats:
            print(f"  α: [{qmf_stats['alpha_0']:.3f}, {qmf_stats['alpha_1']:.3f}, {qmf_stats['alpha_2']:.3f}]")
            print(f"  β: [{qmf_stats['beta_0']:.3f}, {qmf_stats['beta_1']:.3f}, {qmf_stats['beta_2']:.3f}]")

        if is_best:
            print(f"  ✓ 最佳模型更新! F1: {best_val_f1:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            cp_path = os.path.join(SAVE_DIR, f'checkpoint-epoch{epoch+1:02d}.pth')
            trainer.save_checkpoint(cp_path, epoch)
            print(f"  ✓ 保存检查点: {cp_path}")

        # 早停
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！验证 F1 在 {EARLY_STOP_PATIENCE} 个 epoch 内未提升")
            print(f"最佳验证 F1: {best_val_f1:.4f} (Epoch {best_epoch + 1})")
            break

        print("-" * 60)

    # ========== 保存最终模型 ==========
    final_path = os.path.join(SAVE_DIR, 'final_model.pth')
    trainer.save_checkpoint(final_path, epoch, best_val_f1)
    print(f"\n最终模型已保存: {final_path}")
    print(f"训练日志: {csv_logger.filepath}")
    print(f"最佳验证 F1: {best_val_f1:.4f} (Epoch {best_epoch + 1})")

    return trainer, best_val_f1, best_epoch


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    trainer, best_f1, best_epoch = train()


if __name__ == '__main__':
    main()