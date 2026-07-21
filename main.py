"""
main.py — 依次训练所有模型，每个模型保存到独立目录
checkpoints/{model_name}/  logs/{model_name}/
"""
import os
import torch
from torch.utils.data import DataLoader

from dataset import SEN12Dataset
from utils.loss import get_seg_loss
from train import SegmentationTrainer

# ===================== 全局配置 =====================
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 50
TRAIN_TXT = 'train.txt'
VAL_TXT = 'val.txt'
DATA_DIR = './data'
JSON_PATH = 'norm.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRADIENT_CLIP = 1.0
PRINT_INTERVAL = 10
SAVE_INTERVAL = 10
NUM_WORKERS = 4
IMG_SIZE = 128
# =====================================================


def build_model(name: str) -> torch.nn.Module:
    """根据名称创建模型实例。"""
    if name == 'UTAE':
        from models.utae import UTAE
        return UTAE(in_channels=15, num_classes=1)

    elif name == 'SwinUTAE':
        from models.swinutae import SwinUNetHeadWithTemporal
        return SwinUNetHeadWithTemporal(
            img_size=IMG_SIZE, in_channels=15, num_classes=1
        )

    elif name == 'ConvGRU':
        from models.convgru import ConvGRU_Seg
        return ConvGRU_Seg(
            num_classes=1, img_res=IMG_SIZE, in_channels=15,
            kernel_size=(3, 3), hidden_dim=16
        )

    elif name == 'UNet3D':
        from models.Unet3d import UNet3D
        return UNet3D(in_channels=15, num_classes=1, img_res=IMG_SIZE)

    elif name == 'CMXSeg':
        from models.CMXSegTemporal import CMXSeg
        return CMXSeg(num_classes=1, img_size=IMG_SIZE)

    elif name == 'CMNextSeg':
        from models.CMNextSegTemporal import CMNextSeg
        return CMNextSeg(num_classes=1, img_size=IMG_SIZE)

    elif name == 'ESASeg':
        from models.ESASegTemporal import CMXSeg as ESASeg
        return ESASeg(num_classes=1, img_size=IMG_SIZE)

    else:
        raise ValueError(f'未知模型: {name}')


def train_one_model(name: str):
    """训练单个模型。"""
    print(f"\n{'#'*60}")
    print(f"#  开始训练: {name}")
    print(f"{'#'*60}")

    save_dir = f'./checkpoints/{name}'
    log_dir = f'./logs/{name}'

    # ---------- DataLoader ----------
    train_dataset = SEN12Dataset(
        txt_path=TRAIN_TXT, data_dir=DATA_DIR,
        json_path=JSON_PATH, augment=True,
    )
    val_dataset = SEN12Dataset(
        txt_path=VAL_TXT, data_dir=DATA_DIR,
        json_path=JSON_PATH, augment=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    # ---------- 模型 ----------
    model = build_model(name)

    # ---------- 优化器 & 调度器 ----------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS,
    )

    # ---------- 训练 ----------
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=get_seg_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        save_dir=save_dir,
        log_dir=log_dir,
        gradient_clip=GRADIENT_CLIP,
        print_interval=PRINT_INTERVAL,
        save_interval=SAVE_INTERVAL,
        pretrained_path=None,
    )
    trainer.train()
    print(f"\n#  {name} 训练完成！")


def main():
    # 训练顺序
    model_names = [
        'UTAE',
        'SwinUTAE',
        'ConvGRU',
        'UNet3D',
        'CMXSeg',
        'CMNextSeg',
        'ESASeg',
    ]

    print(f"设备: {DEVICE}")
    print(f"训练模型列表: {model_names}")
    print(f"保存目录: checkpoints/<model_name>/")
    print(f"日志目录: logs/<model_name>/")
    print(f"总 Epochs: {EPOCHS}")

    for name in model_names:
        try:
            train_one_model(name)
        except Exception as e:
            print(f"\n[ERROR] {name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            print(f'[INFO] 跳过 {name}，继续下一个模型...\n')
            continue


if __name__ == '__main__':
    main()
