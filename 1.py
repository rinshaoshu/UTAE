import os
import random
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class SampleSplitter:
    """
    从txt文件中读取样本，为每个样本随机删除一个或两个数据源，
    所有处理后的样本都输出到同一个文件夹
    """

    def __init__(self, data_dir, output_dir, txt_path, band_config, copy_mode='copy'):
        """
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            txt_path: test.txt文件路径
            band_config: 波段配置，例如 [{'dir': 's2'}, {'dir': 's1'}, {'dir': 's3'}]
            copy_mode: 'copy' 复制文件, 'symlink' 创建软链接
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.txt_path = txt_path
        self.band_config = band_config
        self.band_names = [config['dir'] for config in band_config]
        self.copy_mode = copy_mode

        # 策略配置
        self.strategies = {
            'delete_one': {'weight': 0.5, 'num_delete': 1},
            'delete_two': {'weight': 0.5, 'num_delete': 2},
        }

        # 读取样本列表
        with open(txt_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]

        # 创建输出目录
        self._create_output_dirs()

        # 初始化统计变量
        self.stats = defaultdict(int)
        self.band_delete_stats = defaultdict(int)
        self.missing_files = []
        self.deletion_records = []

    def _create_output_dirs(self):
        """创建输出目录"""
        # 主输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 为每个band创建子目录
        for band in self.band_names:
            (self.output_dir / band).mkdir(parents=True, exist_ok=True)

        # 创建元数据目录（记录删除信息）
        self.meta_dir = self.output_dir / '_deletion_info'
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _copy_file(self, src_path, dst_path):
        """复制文件或创建软链接"""
        if self.copy_mode == 'symlink':
            # 创建软链接（节省空间）
            try:
                os.symlink(src_path, dst_path)
            except FileExistsError:
                pass
        else:
            # 复制文件
            shutil.copy2(src_path, dst_path)

    def _get_strategy(self):
        """根据权重随机选择策略"""
        strategy_names = list(self.strategies.keys())
        weights = [self.strategies[name]['weight'] for name in strategy_names]
        return random.choices(strategy_names, weights=weights, k=1)[0]

    def process(self, seed=42, verbose=True):
        """处理所有样本"""
        random.seed(seed)

        if verbose:
            print(f"总共 {len(self.samples)} 个样本")
            print(f"Bands: {self.band_names}")
            print(f"策略权重: {self.strategies}")
            print(f"复制模式: {self.copy_mode}")
            print(f"输出目录: {self.output_dir}")
            print("\n开始处理...")

        for sample in tqdm(self.samples, desc="处理样本"):
            try:
                # 选择策略
                strategy_name = self._get_strategy()
                num_delete = self.strategies[strategy_name]['num_delete']

                # 选择要删除的bands
                if num_delete > 0 and num_delete < len(self.band_names):
                    delete_bands = random.sample(self.band_names, num_delete)
                else:
                    delete_bands = []

                # 检查哪些文件存在
                available_bands = []
                for band in self.band_names:
                    src_path = self.data_dir / band / sample
                    if src_path.exists():
                        available_bands.append(band)

                # 记录删除信息
                record = {
                    'sample': sample,
                    'strategy': strategy_name,
                    'deleted_bands': delete_bands,
                    'available_bands': available_bands
                }
                self.deletion_records.append(record)

                # 复制文件到主输出目录
                for band in self.band_names:
                    src_path = self.data_dir / band / sample
                    dst_path = self.output_dir / band / sample

                    if src_path.exists() and band not in delete_bands:
                        # 复制或链接文件
                        self._copy_file(src_path, dst_path)
                        self.stats['copied_files'] += 1
                    elif band in delete_bands:
                        # 记录被删除的band
                        self.band_delete_stats[f"{strategy_name}_{band}"] += 1
                        self.stats['deleted_files'] += 1
                    elif not src_path.exists() and band not in delete_bands:
                        # 文件不存在但本应保留
                        self.missing_files.append(f"{sample} - {band}")
                        self.stats['missing_files'] += 1

                self.stats[strategy_name] += 1

            except Exception as e:
                print(f"处理样本 {sample} 时出错: {e}")
                self.stats['failed'] += 1

        # 保存删除信息记录
        self._save_deletion_records()

        # 保存统计信息
        self._save_stats()

        if verbose:
            self._print_stats()

    def _save_deletion_records(self):
        """保存每个样本的删除记录"""
        records_file = self.meta_dir / 'deletion_records.json'
        with open(records_file, 'w') as f:
            json.dump(self.deletion_records, f, indent=2)

        # 也保存一个CSV格式方便查看
        csv_file = self.meta_dir / 'deletion_records.csv'
        with open(csv_file, 'w') as f:
            f.write("sample,strategy,deleted_bands,available_bands\n")
            for record in self.deletion_records:
                f.write(f"{record['sample']},{record['strategy']},"
                        f"{'|'.join(record['deleted_bands'])},"
                        f"{'|'.join(record['available_bands'])}\n")

    def _save_stats(self):
        """保存统计信息"""
        stats_file = self.output_dir / '_processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'total_samples': len(self.samples),
                'band_config': self.band_config,
                'strategies': self.strategies,
                'copy_mode': self.copy_mode,
                'statistics': dict(self.stats),
                'band_delete_stats': dict(self.band_delete_stats),
                'missing_files_count': len(self.missing_files),
                'missing_files': self.missing_files[:100],  # 只保存前100个
                'output_dir': str(self.output_dir)
            }, f, indent=2)

    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("处理完成！统计信息：")
        print("=" * 60)

        total_processed = self.stats.get('delete_one', 0) + self.stats.get('delete_two', 0)
        print(f"\n📊 总体统计:")
        print(f"  - 总样本数: {len(self.samples)}")
        print(f"  - 成功处理: {total_processed} 个样本")
        print(f"  - 失败: {self.stats.get('failed', 0)} 个样本")

        print(f"\n📈 策略分布:")
        for strategy_name in ['delete_one', 'delete_two']:
            count = self.stats.get(strategy_name, 0)
            if total_processed > 0:
                percentage = count / total_processed * 100
                print(f"  - {strategy_name}: {count} 个样本 ({percentage:.1f}%)")

        print(f"\n📁 文件统计:")
        print(f"  - 复制的文件: {self.stats.get('copied_files', 0)}")
        print(f"  - 删除的文件: {self.stats.get('deleted_files', 0)}")
        print(f"  - 缺失的文件: {self.stats.get('missing_files', 0)}")

        if self.band_delete_stats:
            print(f"\n🗑️ 删除的band统计:")
            for key, count in sorted(self.band_delete_stats.items()):
                print(f"  - {key}: {count} 次")

        if self.missing_files:
            print(f"\n⚠️ 警告: {len(self.missing_files)} 个原始文件缺失")
            print("示例缺失文件:")
            for missing in self.missing_files[:10]:
                print(f"    - {missing}")

        print(f"\n📂 输出目录: {self.output_dir}")
        print(f"   - 数据文件: {self.output_dir}/[band_name]/")
        print(f"   - 删除记录: {self.output_dir}/_deletion_info/")
        print("=" * 60)

    def generate_summary_report(self):
        """生成一个简要的报告"""
        report_path = self.output_dir / '_README.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("样本分割数据集说明\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"原始样本总数: {len(self.samples)}\n")
            f.write(f"波段配置: {', '.join(self.band_names)}\n")
            f.write(f"复制模式: {self.copy_mode}\n\n")
            f.write("数据说明:\n")
            f.write("- 每个样本随机删除了1个或2个波段的数据\n")
            f.write("- 删除的波段没有对应的数据文件\n")
            f.write("- 详细信息请查看 _deletion_info/deletion_records.csv\n\n")
            f.write("目录结构:\n")
            f.write("- [band_name]/: 各波段的数据文件（只包含保留的文件）\n")
            f.write("- _deletion_info/: 删除信息的元数据\n")
            f.write("  - deletion_records.json: 完整删除记录\n")
            f.write("  - deletion_records.csv: CSV格式的删除记录\n")
            f.write("- _processing_stats.json: 处理统计信息\n")

        # 生成一个简单的样本列表文件
        sample_list_file = self.output_dir / '_sample_list.txt'
        with open(sample_list_file, 'w') as f:
            for record in self.deletion_records:
                f.write(f"{record['sample']}\t{record['strategy']}\t{','.join(record['deleted_bands'])}\n")


# ==================== 配置区域 ====================
# 在这里修改你的配置

# 原始数据目录
DATA_DIR = "data"

# 输出目录（新创建的文件夹）
OUTPUT_DIR = "miss_modality"

# test.txt文件路径
TXT_PATH = "path/test.txt"

# 波段配置（根据你的实际目录修改）
BAND_CONFIG = [
    {'dir': 's2'},  # Sentinel-2
    {'dir': 'asc'},  # Sentinel-1
    {'dir': 'dsc'},  # Sentinel-3
]

# 复制模式: 'copy' 复制文件, 'symlink' 创建软链接（节省磁盘空间）
COPY_MODE = 'copy'  # 或 'symlink'

# 随机种子（保证可重复性）
RANDOM_SEED = 42


# ==================== 主程序 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("样本分割工具 - 创建部分数据集（所有样本在一个文件夹）")
    print("=" * 60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 样本列表: {TXT_PATH}")
    print(f"📊 波段配置: {[cfg['dir'] for cfg in BAND_CONFIG]}")
    print(f"📋 复制模式: {COPY_MODE}")
    print(f"🎲 随机种子: {RANDOM_SEED}")
    print("=" * 60)

    # 检查输入文件是否存在
    if not os.path.exists(TXT_PATH):
        print(f"❌ 错误: 找不到文件 {TXT_PATH}")
        return

    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到数据目录 {DATA_DIR}")
        return

    # 创建分割器
    splitter = SampleSplitter(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        txt_path=TXT_PATH,
        band_config=BAND_CONFIG,
        copy_mode=COPY_MODE
    )

    # 处理样本
    splitter.process(seed=RANDOM_SEED)

    # 生成报告
    splitter.generate_summary_report()

    print(f"\n✅ 处理完成！")
    print(f"📂 查看结果: {OUTPUT_DIR}")
    print(f"📋 查看删除记录: {OUTPUT_DIR}/_deletion_info/deletion_records.csv")


if __name__ == "__main__":
    main()