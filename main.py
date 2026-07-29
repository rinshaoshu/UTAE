import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
import numpy as np
import csv
from datetime import datetime

from dataset.dataset import SEN12Dataset
from models.Unet3d import UNet3D
from models.QMF import QMF
from utils.loss import CE_Dice

# ===================== 配置区 =====================
BATCH_SIZE = 2
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 100
TRAIN_TXT = 'train.txt'
VAL_TXT = 'val.txt'
TEST_TXT = 'test.txt'
DATA_DIR = './data'
JSON_PATH = 'dataset/norm(s2-a-d).json'
BAND = ['s2','asc','dsc']
SAVE_DIR = './checkpoints'
LOG_DIR = './logs'
GRADIENT_CLIP = 1.0
NUM_WORKERS = 4
PRETRAINED_PATH = None
CHECKPOINT_DIR = ''

MODE = 'both'
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MONITOR = 'val_f1'#'val_f1', 'val_iou', 'val_precision', 'val_recall'
EARLY_STOP_MIN_DELTA = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


MODEL = QMF()


class CSVLogger:
    """简单的CSV日志记录器"""

    def __init__(self, log_dir, filename='training_log.csv'):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = [
            'epoch',
            'train_loss', 'train_f1', 'train_iou', 'train_precision', 'train_recall',
            'val_loss', 'val_f1', 'val_iou', 'val_precision', 'val_recall',
            'learning_rate', 'best_val_f1'
        ]

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)

        # 初始化CSV文件，写入表头
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_epoch(self, epoch, train_metrics, val_metrics, lr, best_val_f1):
        """记录一个epoch的数据"""
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
            'best_val_f1': best_val_f1
        }

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class SegmentationModel:
    def __init__(
            self,
            model,
            learning_rate=1e-3,
            weight_decay=0.01,
            gradient_clip=1.0,
            is_fusion=False,
            pretrained_path=None,
            device=DEVICE
    ):
        self.device = device
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.is_fusion = is_fusion

        self.criterion = CE_Dice()
        self.num_classes = 2

        # 初始化指标
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

        self.test_metrics = MetricCollection({
            'iou': MulticlassJaccardIndex(num_classes=self.num_classes, average='macro').to(device),
            'f1': MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device),
            'precision': MulticlassPrecision(num_classes=self.num_classes, average='macro').to(device),
            'recall': MulticlassRecall(num_classes=self.num_classes, average='macro').to(device),
        })

        self.train_loss = MeanMetric().to(device)
        self.val_loss = MeanMetric().to(device)
        self.test_loss = MeanMetric().to(device)

        self.optimizer = None
        self.scheduler = None

        self._load_pretrained(pretrained_path)
        self._setup_optimizer()

    def _load_pretrained(self, pretrained_path):
        if pretrained_path is None:
            print("未指定预训练权重，从头开始训练")
            return

        if not os.path.exists(pretrained_path):
            print(f"警告：预训练权重文件不存在: {pretrained_path}，从头开始训练")
            return

        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"警告：以下权重层未从预训练模型中加载: {missing_keys}")
            if unexpected_keys:
                print(f"警告：预训练模型中有以下额外权重层: {unexpected_keys}")

            print(f"✓ 成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"✗ 加载预训练权重失败: {e}")
            print("继续从头开始训练")

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

    def forward(self, x):
        if self.is_fusion:
            return self.model(x, train=self.training)
        return self.model(x)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.train_loss.reset()
        self.train_metrics.reset()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]')

        for batch_idx, (data, mask) in enumerate(pbar):
            data = data.to(self.device)
            mask = mask.to(self.device)

            # 前向传播
            output = self.forward(data)
            loss = self.criterion(output, mask)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # 更新指标
            self.train_loss.update(loss)
            mask_indices = torch.argmax(mask, dim=1)
            pred_indices = torch.argmax(output, dim=1)
            self.train_metrics.update(pred_indices, mask_indices)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 计算epoch指标
        avg_loss = self.train_loss.compute()
        metrics = self.train_metrics.compute()
        metrics = {f'train_{k}': v.item() for k, v in metrics.items()}
        metrics['train_loss'] = avg_loss.item()

        self.train_loss.reset()
        self.train_metrics.reset()

        return metrics

    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        self.val_loss.reset()
        self.val_metrics.reset()

        pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Val]')

        for data, mask in pbar:
            data = data.to(self.device)
            mask = mask.to(self.device)

            output = self.forward(data)
            loss = self.criterion(output, mask)

            self.val_loss.update(loss)
            mask_indices = torch.argmax(mask, dim=1)
            pred_indices = torch.argmax(output, dim=1)
            self.val_metrics.update(pred_indices, mask_indices)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = self.val_loss.compute()
        metrics = self.val_metrics.compute()
        metrics = {f'val_{k}': v.item() for k, v in metrics.items()}
        metrics['val_loss'] = avg_loss.item()

        self.val_loss.reset()
        self.val_metrics.reset()

        return metrics

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        self.test_loss.reset()
        self.test_metrics.reset()

        pbar = tqdm(test_loader, desc='Testing')

        for data, mask in pbar:
            data = data.to(self.device)
            mask = mask.to(self.device)

            output = self.forward(data)
            loss = self.criterion(output, mask)

            self.test_loss.update(loss)
            mask_indices = torch.argmax(mask, dim=1)
            pred_indices = torch.argmax(output, dim=1)
            self.test_metrics.update(pred_indices, mask_indices)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = self.test_loss.compute()
        metrics = self.test_metrics.compute()
        metrics = {f'test_{k}': v.item() for k, v in metrics.items()}
        metrics['test_loss'] = avg_loss.item()

        self.test_loss.reset()
        self.test_metrics.reset()

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

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('best_f1', None)


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

    print("加载测试数据集...")
    test_dataset = SEN12Dataset(
        txt_path=TEST_TXT,
        data_dir=DATA_DIR,
        norm_dir=JSON_PATH,
        band=BAND,
        augment=False,
    )
    print(f"测试集大小: {len(test_dataset)}")

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train():
    """训练和验证循环"""
    # 创建模型
    model = MODEL

    # 创建训练器
    trainer = SegmentationModel(
        model=model,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        is_fusion=False,
        pretrained_path=PRETRAINED_PATH,
        device=DEVICE
    )

    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders()

    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 创建CSV日志记录器
    csv_logger = CSVLogger(LOG_DIR, 'training_log.csv')

    # 早停
    best_val_f1 = -float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\n开始训练，共 {EPOCHS} 个epoch")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # 验证
        val_metrics = trainer.validate_epoch(val_loader, epoch)

        # 更新学习率
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # 检查是否是最佳模型
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            best_epoch = epoch
            patience_counter = 0

            checkpoint_path = os.path.join(
                SAVE_DIR,
                f'best_model-epoch{epoch + 1:02d}-f1{best_val_f1:.4f}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, best_val_f1)
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        # 记录到CSV
        csv_logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, best_val_f1)

        # 打印信息
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train Loss: {train_metrics['train_loss']:.4f}, F1: {train_metrics['train_f1']:.4f}, IoU: {train_metrics['train_iou']:.4f}")
        print(
            f"Val Loss: {val_metrics['val_loss']:.4f}, F1: {val_metrics['val_f1']:.4f}, IoU: {val_metrics['val_iou']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        if is_best:
            print(f"✓ 保存最佳模型: {checkpoint_path}")
            print(f"  最佳验证F1: {best_val_f1:.4f}")

        # 保存定期检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                SAVE_DIR,
                f'checkpoint-epoch{epoch + 1:02d}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch)
            print(f"✓ 保存检查点: {checkpoint_path}")

        # 早停检查
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！验证F1在{EARLY_STOP_PATIENCE}个epoch内未提升")
            print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch + 1})")
            break

        print("-" * 60)

    # 保存最终模型
    final_path = os.path.join(SAVE_DIR, 'final_model.pth')
    trainer.save_checkpoint(final_path, epoch, best_val_f1)
    print(f"\n最终模型已保存到: {final_path}")

    print(f"\n训练日志已保存到: {csv_logger.filepath}")

    return trainer, best_val_f1, best_epoch


def test_model(test_loader, checkpoint_path=CHECKPOINT_DIR):
    """测试模型"""
    print("\n开始测试...")

    # 创建模型
    model = MODEL

    trainer = SegmentationModel(
        model=model,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        is_fusion=False,
        device=DEVICE
    )

    # 加载checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    else:
        # 尝试加载最佳模型
        best_model_path = None
        if os.path.exists(SAVE_DIR):
            for f in os.listdir(SAVE_DIR):
                if f.startswith('best_model') and f.endswith('.pth'):
                    best_model_path = os.path.join(SAVE_DIR, f)
                    break

        if best_model_path:
            print(f"加载最佳模型: {best_model_path}")
            trainer.load_checkpoint(best_model_path)
        else:
            print("未找到模型文件，使用当前模型进行测试")

    # 测试
    test_metrics = trainer.test(test_loader)

    # 保存测试结果到CSV
    test_csv_path = os.path.join(LOG_DIR, 'test_results.csv')
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in test_metrics.items():
            writer.writerow([key, f'{value:.4f}'])

    print("\n" + "=" * 60)
    print("测试结果:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)
    print(f"\n测试结果已保存到: {test_csv_path}")

    return test_metrics


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders()

    if MODE in ('train', 'both'):
        # 训练
        trainer, best_f1, best_epoch = train()

        # 训练完成后自动测试
        if MODE == 'both':
            print("\n训练完成，开始测试最佳模型...")
            # 加载最佳模型进行测试
            best_model_path = None
            if os.path.exists(SAVE_DIR):
                for f in os.listdir(SAVE_DIR):
                    if f.startswith('best_model') and f.endswith('.pth'):
                        best_model_path = os.path.join(SAVE_DIR, f)
                        break

            if best_model_path:
                test_model(test_loader, best_model_path)
            else:
                test_model(test_loader)

    elif MODE == 'test':
        # 仅测试
        test_model(test_loader)


if __name__ == '__main__':
    main()