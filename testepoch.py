"""
test0001.py — 所有模型各跑 3 个 training batch，验证 forward/loss/backward 全流程
"""
import torch
from torch.utils.data import DataLoader, Subset

from dataset import SEN12Dataset
from utils.loss import CE_Dice, DSTLoss

# ===================== 配置 =====================
BATCH_SIZE = 4
NUM_BATCHES = 3
IMG_SIZE = 128
TRAIN_TXT = 'train.txt'
DATA_DIR = './data'
JSON_PATH = 'norm.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =================================================


def build_model(name):
    """复用 check.py 的模型构建逻辑。"""
    if name == 'UTAE':
        from models.utae import UTAE
        return UTAE(in_channels=15, num_classes=2)

    elif name == 'SwinUTAE':
        from models.swinutae import SwinUNetHeadWithTemporal
        return SwinUNetHeadWithTemporal(
            img_size=IMG_SIZE, in_channels=15, num_classes=2
        )

    elif name == 'ConvGRU':
        from models.convgru import ConvGRU_Seg
        return ConvGRU_Seg(
            num_classes=2, img_res=IMG_SIZE, in_channels=15,
            kernel_size=(3, 3), hidden_dim=16
        )

    elif name == 'UNet3D':
        from models.Unet3d import UNet3D
        return UNet3D(in_channels=15, num_classes=2, img_res=IMG_SIZE)

    elif name == 'CMXSeg':
        from models.CMXSegTemporal import CMXSeg
        return CMXSeg(num_classes=2, img_size=IMG_SIZE)

    elif name == 'CMNextSeg':
        from models.CMNextSegTemporal import CMNextSeg
        return CMNextSeg(num_classes=2, img_size=IMG_SIZE)

    elif name == 'ESASeg':
        from models.ESASegTemporal import CMXSeg as ESASeg
        return ESASeg(num_classes=2, img_size=IMG_SIZE)

    elif name == 'DSTFusion':
        from models.DSTUtea import nnFormerDSTemporalFusion
        return nnFormerDSTemporalFusion(
            num_classes=2, crop_size=[IMG_SIZE, IMG_SIZE],
        )

    else:
        raise ValueError(f'未知模型: {name}')


def train_3batches(name, model, loader_iter):
    """对单个模型跑 3 个 training batch（forward + loss + backward + step）。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    is_fusion = (name == 'DSTFusion')
    criterion = DSTLoss() if is_fusion else CE_Dice()

    batch_losses = []
    for i in range(NUM_BATCHES):
        data, mask = next(loader_iter)
        data, mask = data.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        if is_fusion:
            output = model(data, train=True)
        else:
            output = model(data)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        print(f"    Batch {i+1}/{NUM_BATCHES}: loss={loss.item():.4f}, "
              f"in={tuple(data.shape)}, out={tuple(output[0].shape if is_fusion else output.shape)}")

    return batch_losses


def main():
    print(f"设备: {DEVICE}")
    print(f"每个模型跑 {NUM_BATCHES} 个 training batch, batch_size={BATCH_SIZE}\n")

    # ---------- DataLoader ----------
    dataset = SEN12Dataset(
        txt_path=TRAIN_TXT, data_dir=DATA_DIR,
        json_path=JSON_PATH, augment=True,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    loader_iter = iter(loader)

    # ---------- 模型列表 ----------
    model_names = [
        'UTAE', 'SwinUTAE', 'ConvGRU', 'UNet3D',
        'CMXSeg', 'CMNextSeg', 'ESASeg', 'DSTFusion',
    ]

    results = {}
    for name in model_names:
        print(f"{'='*55}")
        print(f"[{name}]")
        try:
            model = build_model(name).to(DEVICE)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {param_count:,}")

            losses = train_3batches(name, model, loader_iter)

            avg_loss = sum(losses) / len(losses)
            print(f"  ✓ 通过  平均 loss: {avg_loss:.4f}")
            results[name] = '✓'

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = '✗'

    # ---------- 汇总 ----------
    print(f"\n{'='*55}")
    print("训练测试汇总:")
    for name in model_names:
        print(f"  {results[name]} {name}")
    passed = sum(1 for v in results.values() if v == '✓')
    print(f"\n通过: {passed}/{len(model_names)}")


if __name__ == '__main__':
    main()
