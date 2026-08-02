import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from dataset.dataset import SEN12Dataset
from model import get_model
from utils.loss import CE_Dice

# ===================== 配置 =====================
# 模型路径 - 在这里修改你要测试的模型
MODEL_PATH = 'checkpoints/Unet_asc/best_model.pth'  # 修改为你的模型路径

# 数据配置
BATCH_SIZE = 8
TEST_TXT = 'path/test.txt'
DATA_DIR = 'data'
JSON_PATH = 'dataset/asc_norm.json'
BAND = [ 'asc']
NUM_WORKERS = 4
OUTPUT_DIR = './test/normal/asc_unet'

# 模型配置
MODEL = get_model()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


class Tester:
    def __init__(self, model):
        self.device = DEVICE
        self.model = model.to(DEVICE)
        self.criterion = CE_Dice()
        self.metrics = MetricCollection({
            'iou': MulticlassJaccardIndex(num_classes=2, average='macro').to(DEVICE),
            'f1': MulticlassF1Score(num_classes=2, average='macro').to(DEVICE),
            'precision': MulticlassPrecision(num_classes=2, average='macro').to(DEVICE),
            'recall': MulticlassRecall(num_classes=2, average='macro').to(DEVICE),
        })
        self.loss = MeanMetric().to(DEVICE)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        print(f"模型加载成功: {path}")

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        self.loss.reset()
        self.metrics.reset()

        for data, mask in tqdm(loader, desc='Testing', leave=False):
            data, mask = data.to(self.device), mask.to(self.device)
            output = self.model(data)

            # 计算损失
            loss = self.criterion(output, mask)
            self.loss.update(loss)

            # 获取预测
            pred = torch.argmax(output, dim=1)
            target = torch.argmax(mask, dim=1)
            self.metrics.update(pred, target)

        # 计算总体指标
        results = {
            'loss': self.loss.compute().item(),
            'f1': self.metrics.compute()['f1'].item(),
            'iou': self.metrics.compute()['iou'].item(),
            'precision': self.metrics.compute()['precision'].item(),
            'recall': self.metrics.compute()['recall'].item(),
        }
        return results


def get_loader():
    dataset = SEN12Dataset(txt_path=TEST_TXT, data_dir=DATA_DIR, norm_dir=JSON_PATH, band=BAND, augment=False)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def test_single_model(model_path, output_dir=None):
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
    print("加载数据集...")
    loader = get_loader()
    print(f"数据集大小: {len(loader.dataset)} 个样本")

    # 创建测试器
    tester = Tester(MODEL)

    # 加载模型
    print(f"加载模型: {model_path}")
    try:
        tester.load_model(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    # 测试模型
    print("开始测试...")
    model_name = os.path.basename(model_path).replace('.pth', '')
    metrics = tester.test(loader)

    # 添加模型信息
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'loss': metrics['loss'],
        'f1': metrics['f1'],
        'iou': metrics['iou'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"测试结果 - {model_name}")
    print(f"{'=' * 60}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"IoU: {results['iou']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"{'=' * 60}\n")

    # 保存结果到CSV（每个指标作为一列）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame([results])
    csv_path = f'{output_dir}/results_{model_name}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"结果已保存: {csv_path}")
    print(f"\nCSV:")
    print(df.to_string(index=False))

    # 保存详细的文本报告
    report_path = f'{output_dir}/report_{model_name}_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"model: {model_name}\n")
        f.write(f"{len(loader.dataset)}\n")
        f.write(f"{'=' * 60}\n")

        f.write(f"  Loss: {results['loss']:.6f}\n")
        f.write(f"  F1 Score: {results['f1']:.6f}\n")
        f.write(f"  IoU: {results['iou']:.6f}\n")
        f.write(f"  Precision: {results['precision']:.6f}\n")
        f.write(f"  Recall: {results['recall']:.6f}\n")
        f.write(f"{'=' * 60}\n")

    print(f"详细报告已保存: {report_path}")

    return results


def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"{'=' * 60}")
    print(f"模型测试工具")
    print(f"{'=' * 60}")
    print(f"测试模型: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    # 测试单个模型
    test_single_model(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR
    )


if __name__ == '__main__':
    main()