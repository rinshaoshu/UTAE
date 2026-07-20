import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SEN12Dataset
from models.utae import UTAE
from models.swinutae import SwinUNetHeadWithTemporal
from models.CMXSegTemporal import CMXSeg
from models.CMNextSegTemporal import CMNextSeg

from utils.loss import get_seg_loss
from utils.metrics import compute_metrics
from utils.logger import CSVLogger

# ===================== 配置区（要改直接改源码） =====================
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 50
TRAIN_TXT = 'train.txt'
VAL_TXT = 'val.txt'
DATA_DIR = './data'
JSON_PATH = 'norm.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = './checkpoints/CMNextSeg'
LOG_DIR = './logs/CMNextSeg'
GRADIENT_CLIP = 1.0
PRINT_INTERVAL = 10
SAVE_INTERVAL = 10
NUM_WORKERS = 4
PRETRAINED_PATH = 'checkpoints/best_model.pth'  # 预训练权重路径，None表示不加载



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
            pretrained_path=None,  # 新增参数
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.gradient_clip = gradient_clip
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.pretrained_path = pretrained_path

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.logger = CSVLogger(log_dir)

        self.best_iou = 0.0
        self.model.to(self.device)

        # 加载预训练权重
        self._load_pretrained()

    def _load_pretrained(self):
        """加载预训练权重（如果有）。"""
        if self.pretrained_path is None:
            print("未指定预训练权重，从头开始训练")
            return

        if not os.path.exists(self.pretrained_path):
            print(f"警告：预训练权重文件不存在: {self.pretrained_path}，从头开始训练")
            return

        try:
            # 加载权重
            state_dict = torch.load(self.pretrained_path, map_location=self.device)

            # 处理可能的 key 不匹配（例如模型保存时带了 'module.' 前缀）
            if list(state_dict.keys())[0].startswith('module.'):
                # 去掉 DataParallel 添加的 'module.' 前缀
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # 加载权重，忽略不匹配的 key（例如分类头尺寸不同）
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"警告：以下权重层未从预训练模型中加载: {missing_keys}")
            if unexpected_keys:
                print(f"警告：预训练模型中有以下额外权重层: {unexpected_keys}")

            print(f"✓ 成功加载预训练权重: {self.pretrained_path}")

        except Exception as e:
            print(f"✗ 加载预训练权重失败: {e}")
            print("继续从头开始训练")

    def _train_one_epoch(self, epoch):
        """训练一个 epoch，返回平均 train_loss。"""
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs} [Train]')
        for batch_idx, (data, mask) in enumerate(pbar):
            data = data.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
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

        all_outputs = torch.cat(all_outputs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        metrics = compute_metrics(all_outputs, all_masks)

        return avg_loss, metrics

    def train(self):
        """完整训练流程。"""
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, metrics = self._validate()
            self.scheduler.step()

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

            if metrics['iou'] > self.best_iou:
                self.best_iou = metrics['iou']
                best_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_path)
                print(f'  → 新最佳 IoU: {self.best_iou:.4f}，已保存 {best_path}')

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
    model = CMNextSeg(num_classes=1, img_size=128, drop_path_rate=0.1)

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=EPOCHS)

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
        pretrained_path=None,  # 传入预训练路径
    )
    trainer.train()


if __name__ == '__main__':
    main()