import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import numpy as np

from dataset.dataset import SEN12Dataset
from models.QMF import QMF, QMFLoss
from utils.loss import CE_Dice

# ===================== 配置 =====================
# 模型路径 - 在这里修改你要测试的模型
MODEL_PATH = 'checkpoints/qmf/best_model.pth'  # 或 'checkpoints/checkpoint-epoch10.pth'

# 数据配置
BATCH_SIZE = 8  # QMF 建议 batch_size=1
TEST_TXT = 'path/test.txt'
DATA_DIR = 'miss_all'
JSON_PATH = 'dataset/norm.json'
BAND = ['s2', 'asc', 'dsc']  # QMF 需要三个模态
NUM_WORKERS = 4
OUTPUT_DIR = './test/EMM/qmf_results'

# QMF 模型参数（必须与训练时一致）
CHANNEL_SPLITS = [11, 3, 3]  # s2, asc, dsc 的输入通道数
DIM = 2  # 通道维度索引
IMG_RES = 128
NUM_CLASSES = 2
DROPOUT = 0.0

# 损失参数
REG_LAMBDA = 0.1
BUFFER_SIZE = 200
USE_REG = True
USE_DICE = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


class QMFTester:
    """QMF 模型测试器"""

    def __init__(self, model, loss_fn=None):
        self.device = DEVICE
        self.model = model.to(DEVICE)

        # 使用 QMF Loss（如果需要正则化损失）
        if loss_fn is None:
            self.loss_fn = QMFLoss(
                num_modalities=3,
                reg_lambda=REG_LAMBDA,
                buffer_size=BUFFER_SIZE,
                use_reg=USE_REG,
                ignore_index=255,
                use_dice=USE_DICE
            ).to(DEVICE)
        else:
            self.loss_fn = loss_fn.to(DEVICE)

        # 指标
        self.metrics = MetricCollection({
            'iou': MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro').to(DEVICE),
            'f1': MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(DEVICE),
            'precision': MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(DEVICE),
            'recall': MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(DEVICE),
        })

        # 损失追踪
        self.total_loss = MeanMetric().to(DEVICE)
        self.fused_loss = MeanMetric().to(DEVICE)
        self.unimodal_loss = MeanMetric().to(DEVICE)
        self.reg_loss = MeanMetric().to(DEVICE)

        # 各模态损失追踪
        self.modality_losses = [MeanMetric().to(DEVICE) for _ in range(3)]

        # 融合权重追踪
        self.avg_weights = []

    def load_model(self, path):
        """加载模型权重"""
        ckpt = torch.load(path, map_location=self.device)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # 处理 DataParallel 权重
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # 加载权重（允许不严格匹配）
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺少的键: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告: 多余的键: {unexpected_keys[:5]}...")

        print(f"模型加载成功: {path}")

        # 如果checkpoint包含epoch和best_f1信息，打印出来
        if 'epoch' in ckpt:
            print(f"  Epoch: {ckpt['epoch'] + 1}")
        if 'best_f1' in ckpt:
            print(f"  Best F1: {ckpt['best_f1']:.4f}")

    @torch.no_grad()
    def test(self, loader, save_predictions=False, save_dir=None):
        """
        测试模型

        Args:
            loader: 数据加载器
            save_predictions: 是否保存预测结果
            save_dir: 预测结果保存目录

        Returns:
            results: 测试结果字典
        """
        self.model.eval()
        self.loss_fn.eval()

        # 重置指标
        self.total_loss.reset()
        self.fused_loss.reset()
        self.unimodal_loss.reset()
        self.reg_loss.reset()
        for m in range(3):
            self.modality_losses[m].reset()
        self.metrics.reset()
        self.avg_weights = []

        # 创建保存目录
        if save_predictions and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)

        pbar = tqdm(loader, desc='Testing', leave=False)

        for batch_idx, (data, mask) in enumerate(pbar):
            data = data.to(self.device)
            mask = mask.to(self.device)  # (B, 2, H, W) one-hot

            # ========== QMF 前向传播 ==========
            # 返回: (fused_logits, logits_list, weights, energies)
            fused_logits, logits_list, weights, energies = self.model(data)
            # fused_logits: (B, 2, H, W)
            # logits_list: [s2_out, asc_out, dsc_out] 每个 (B, 2, H, W)
            # weights: (B, 3, H, W)
            # energies: (B, 3, H, W)

            # mask 转为索引格式
            mask_indices = mask.argmax(dim=1)  # (B, H, W)

            # ========== 计算损失 ==========
            total_loss, loss_dict = self.loss_fn(
                fused_logits,
                logits_list,
                mask_indices,
                weights
            )

            # ========== 更新损失 ==========
            self.total_loss.update(total_loss)
            self.fused_loss.update(loss_dict['fused'])
            self.unimodal_loss.update(loss_dict['unimodal'])
            self.reg_loss.update(loss_dict['reg'])

            # 更新各模态损失
            for m in range(3):
                self.modality_losses[m].update(loss_dict['per_modality'][m])

            # ========== 更新分割指标 ==========
            pred_indices = fused_logits.argmax(dim=1)  # (B, H, W)
            self.metrics.update(pred_indices, mask_indices)

            # ========== 记录融合权重 ==========
            if weights is not None:
                avg_weight = weights.mean(dim=[0, 2, 3]).cpu().numpy()  # (3,)
                self.avg_weights.append(avg_weight)

            # ========== 保存预测结果 ==========
            if save_predictions and save_dir is not None:
                # 保存预测图 (仅保存第一个batch的第一个样本)
                if batch_idx == 0:
                    self._save_predictions(
                        data[0], mask[0], fused_logits[0],
                        weights[0] if weights is not None else None,
                        energies[0] if energies is not None else None,
                        save_dir
                    )

            # ========== 更新进度条 ==========
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'f1': f'{self.metrics.compute()["f1"].item():.4f}'
            })

        # ========== 计算总体指标 ==========
        metrics = self.metrics.compute()

        results = {
            'total_loss': self.total_loss.compute().item(),
            'fused_loss': self.fused_loss.compute().item(),
            'unimodal_loss': self.unimodal_loss.compute().item(),
            'reg_loss': self.reg_loss.compute().item(),
            'f1': metrics['f1'].item(),
            'iou': metrics['iou'].item(),
            'precision': metrics['precision'].item(),
            'recall': metrics['recall'].item(),
        }

        # 各模态损失
        for m in range(3):
            results[f'modality_{m}_loss'] = self.modality_losses[m].compute().item()

        # 平均融合权重
        if self.avg_weights:
            avg_weights = np.mean(self.avg_weights, axis=0)
            for m in range(3):
                results[f'weight_{m}'] = float(avg_weights[m])

        return results

    def _save_predictions(self, data, mask, fused_logits, weights, energies, save_dir):
        """保存预测结果示例"""
        try:
            import matplotlib.pyplot as plt

            # 转换为numpy
            data_np = data.cpu().numpy()  # (T, C, H, W)
            mask_np = mask.cpu().numpy()  # (2, H, W)
            pred = fused_logits.argmax(dim=0).cpu().numpy()  # (H, W)

            # 创建图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 显示各模态的波段（取第一个时间步的第一个波段）
            # S2 (RGB合成)
            if data_np.shape[1] >= 11:
                s2_rgb = data_np[0, [3, 2, 1], :, :]  # B4, B3, B2 (RGB)
                s2_rgb = (s2_rgb - s2_rgb.min()) / (s2_rgb.max() - s2_rgb.min() + 1e-8)
                axes[0, 0].imshow(s2_rgb.transpose(1, 2, 0))
                axes[0, 0].set_title('S2 (RGB)')
                axes[0, 0].axis('off')

            # ASC
            if data_np.shape[1] >= 14:
                asc = data_np[0, 11, :, :]
                axes[0, 1].imshow(asc, cmap='gray')
                axes[0, 1].set_title('ASC')
                axes[0, 1].axis('off')

            # DSC
            if data_np.shape[1] >= 17:
                dsc = data_np[0, 14, :, :]
                axes[0, 2].imshow(dsc, cmap='gray')
                axes[0, 2].set_title('DSC')
                axes[0, 2].axis('off')

            # Ground Truth
            gt = mask_np.argmax(axis=0)  # (H, W)
            axes[1, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')

            # Prediction
            axes[1, 1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title('Prediction')
            axes[1, 1].axis('off')

            # Fusion Weights
            if weights is not None:
                weights_np = weights.cpu().numpy()  # (3, H, W)
                # 显示平均权重
                avg_weight = weights_np.mean(axis=0)  # (H, W)
                im = axes[1, 2].imshow(avg_weight, cmap='hot')
                axes[1, 2].set_title(
                    f'Avg Weight (S2={weights_np[0].mean():.2f}, ASC={weights_np[1].mean():.2f}, DSC={weights_np[2].mean():.2f})')
                axes[1, 2].axis('off')
                plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'predictions', 'sample.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"预测示例已保存: {os.path.join(save_dir, 'predictions', 'sample.png')}")

        except Exception as e:
            print(f"保存预测示例失败: {e}")


def get_loader():
    """获取测试数据加载器"""
    print("加载测试数据集...")
    dataset = SEN12Dataset(
        txt_path=TEST_TXT,
        data_dir=DATA_DIR,
        norm_dir=JSON_PATH,
        band=BAND,
        augment=False,
    )
    print(f"测试集大小: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return loader


def test_single_model(model_path, output_dir=None, save_predictions=True):
    """测试单个模型并保存结果"""

    # 忽略 OMP_NUM_THREADS 警告
    os.environ['OMP_NUM_THREADS'] = '1'

    if output_dir is None:
        output_dir = OUTPUT_DIR

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取数据加载器
    loader = get_loader()

    # ========== 创建 QMF 模型 ==========
    model = QMF(
        channel_splits=CHANNEL_SPLITS,
        dim=DIM,
        img_res=IMG_RES,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    )

    # ========== 创建损失函数 ==========
    loss_fn = QMFLoss(
        num_modalities=3,
        reg_lambda=REG_LAMBDA,
        buffer_size=BUFFER_SIZE,
        use_reg=USE_REG,
        ignore_index=255,
        use_dice=USE_DICE
    )

    # ========== 创建测试器 ==========
    tester = QMFTester(model, loss_fn)

    # ========== 加载模型 ==========
    print(f"加载模型: {model_path}")
    try:
        tester.load_model(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ========== 测试模型 ==========
    print("开始测试...")
    model_name = os.path.basename(model_path).replace('.pth', '')

    # 创建预测保存目录
    save_dir = os.path.join(output_dir, model_name) if save_predictions else None

    metrics = tester.test(
        loader,
        save_predictions=save_predictions,
        save_dir=save_dir
    )

    # ========== 整理结果 ==========
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **metrics
    }

    # ========== 打印结果 ==========
    print(f"\n{'=' * 70}")
    print(f"QMF 模型测试结果 - {model_name}")
    print(f"{'=' * 70}")
    print(f"Loss:")
    print(f"  Total Loss: {results['total_loss']:.4f}")
    print(f"  Fused Loss: {results['fused_loss']:.4f}")
    print(f"  Unimodal Loss: {results['unimodal_loss']:.4f}")
    print(f"  Reg Loss: {results['reg_loss']:.4f}")
    print(f"\nSegmentation Metrics:")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"\nPer Modality Loss:")
    for m in range(3):
        print(f"  Modality {m}: {results[f'modality_{m}_loss']:.4f}")
    print(f"\nFusion Weights:")
    for m in range(3):
        if f'weight_{m}' in results:
            print(f"  Modality {m}: {results[f'weight_{m}']:.4f}")
    print(f"{'=' * 70}\n")

    # ========== 保存结果到CSV ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame([results])
    csv_path = os.path.join(output_dir, f'results_{model_name}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"结果已保存: {csv_path}")

    # ========== 保存详细的文本报告 ==========
    report_path = os.path.join(output_dir, f'report_{model_name}_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"QMF 模型测试报告\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Time: {results['timestamp']}\n")
        f.write(f"Test Samples: {len(loader.dataset)}\n")
        f.write(f"{'=' * 70}\n\n")

        f.write(f"LOSS:\n")
        f.write(f"  Total Loss: {results['total_loss']:.6f}\n")
        f.write(f"  Fused Loss: {results['fused_loss']:.6f}\n")
        f.write(f"  Unimodal Loss: {results['unimodal_loss']:.6f}\n")
        f.write(f"  Reg Loss: {results['reg_loss']:.6f}\n\n")

        f.write(f"SEGMENTATION METRICS:\n")
        f.write(f"  F1 Score: {results['f1']:.6f}\n")
        f.write(f"  IoU: {results['iou']:.6f}\n")
        f.write(f"  Precision: {results['precision']:.6f}\n")
        f.write(f"  Recall: {results['recall']:.6f}\n\n")

        f.write(f"PER MODALITY LOSS:\n")
        for m in range(3):
            f.write(f"  Modality {m}: {results[f'modality_{m}_loss']:.6f}\n")

        if any(f'weight_{m}' in results for m in range(3)):
            f.write(f"\nFUSION WEIGHTS:\n")
            for m in range(3):
                if f'weight_{m}' in results:
                    f.write(f"  Modality {m}: {results[f'weight_{m}']:.6f}\n")

        f.write(f"\n{'=' * 70}\n")

    print(f"详细报告已保存: {report_path}")

    return results


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"{'=' * 70}")
    print(f"QMF 模型测试工具")
    print(f"{'=' * 70}")
    print(f"测试模型: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print(f"{'=' * 70}\n")

    # 测试单个模型
    test_single_model(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        save_predictions=True
    )


if __name__ == '__main__':
    main()