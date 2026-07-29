import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import json
from .transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotate90,
    Normalize,
    Compose,

)
from .split.bands import get_bands_by_dirs

class SEN12Dataset(Dataset):
    def __init__(self, txt_path, data_dir=None ,norm_dir=None,band=None, augment=False):
        self.txt_path = txt_path
        self.data_dir = str(data_dir)
        self.augment = augment
        self.norm_dir = norm_dir
        # 配置：文件夹 -> 波段列表
        self.bands_config = get_bands_by_dirs(band)

        # 读取文件名（不过滤）
        with open(txt_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]

        self.all_bands = []
        for config in self.bands_config:
            self.all_bands.extend(config['bands'])

        # ---- 预计算文件路径和 mask 路径，过滤不存在的文件 ----
        self.sample_paths = []  # list[list[str]] 每个样本对应每个 satellite dir 的完整路径
        self.sample_exists = [] # list[list[bool]] 每个样本每个路径是否存在
        self.mask_paths = []    # list[str] 每个样本第一个可用的 mask 路径
        valid_file_list = []    # 只保留实际存在的文件
        missing_count = 0

        for fname in self.file_list:
            paths = []
            exists = []
            for config in self.bands_config:
                p = os.path.join(self.data_dir, config['dir'], fname)
                paths.append(p)
                exists.append(os.path.isfile(p))

            # 检查是否至少有一个目录中存在该文件
            if not any(exists):
                missing_count += 1
                continue  # 跳过不存在的文件

            self.sample_paths.append(paths)
            self.sample_exists.append(exists)
            valid_file_list.append(fname)

            # 找 mask 文件的路径（按 bands_config 顺序优先）
            for i, config in enumerate(self.bands_config):
                if exists[i]:
                    self.mask_paths.append(paths[i])
                    break

        if missing_count > 0:
            print(f"[SEN12Dataset] 警告: {missing_count}/{len(self.file_list)} 个文件未找到，已跳过。")
            print(f"  数据目录: {self.data_dir}")
            for cfg in self.bands_config:
                print(f"  检查路径: {os.path.join(self.data_dir, cfg['dir'])}/")

        if len(valid_file_list) == 0:
            raise FileNotFoundError(
                f"[SEN12Dataset] 没有找到任何有效文件！\n"
                f"  txt文件: {txt_path}\n"
                f"  数据目录: {self.data_dir}\n"
                f"  请检查数据路径是否正确。"
            )

        self.file_list = valid_file_list

        # 只创建一次 transform（避免每次 __getitem__ 重复读 JSON / 创建对象）
        if self.augment:
            self.transform = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(self.norm_dir)
            ])
        else:
            self.transform = Compose([Normalize(self.norm_dir)])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        all_data = []
        paths = self.sample_paths[idx]
        exists = self.sample_exists[idx]

        # 尝试多种引擎打开 NetCDF 文件
        def _open_nc(filepath):
            for engine in ['h5netcdf', 'netcdf4', 'scipy']:
                try:
                    return xr.open_dataset(filepath, engine=engine)
                except Exception:
                    continue
            raise ValueError(f"Cannot open {filepath} with any backend. "
                             f"Please install h5netcdf: pip install h5netcdf")

        for i, config in enumerate(self.bands_config):
            file_path = paths[i]

            if exists[i]:
                with _open_nc(file_path) as ds:
                    ds = ds.sortby("time")
                    data = np.stack([ds[band].values for band in config['bands']], axis=0)
                    data = torch.from_numpy(data).float().permute(1, 0, 2, 3)
                    all_data.append(data)
            else:
                num_time_steps = 15
                num_bands = len(config['bands'])
                placeholder = torch.zeros(num_time_steps, num_bands, 128, 128)
                all_data.append(placeholder)

        data = torch.cat(all_data, dim=1)

        # mask 路径已在 __init__ 预计算
        mask_path = self.mask_paths[idx]
        with _open_nc(mask_path) as ds:
            mask_data = ds['MASK'].isel(time=0).values
            mask_data = (mask_data > 0).astype(np.int64)
            mask = torch.from_numpy(mask_data).long()

        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # one-hot: 通道0=无效区域, 通道1=有效区域 → (2, H, W)
        mask_one_hot = torch.zeros(2, mask.shape[0], mask.shape[1], dtype=torch.float32)
        mask_one_hot[0, :, :] = (mask == 0).float()
        mask_one_hot[1, :, :] = (mask == 1).float()

        data, mask_one_hot = self.transform(data, mask_one_hot)
        return data, mask_one_hot

# ==================== 使用示例 ====================
from torch.utils.data import DataLoader

# ==================== 使用示例 ====================
# 假设 SEN12Dataset 类已经定义好了
if __name__ == '__main__':
# 1. 创建数据集
    dataset = SEN12Dataset(
        txt_path="split/s2/train.txt",  # 包含文件名的txt
        data_dir="../data/",  # 数据目录
        norm_dir="s2_norm.json",
        band= ['s2'],
        augment=True  # 是否启用数据增强
    )

    # 2. 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 一批4个样本
        shuffle=True,  # 打乱数据
        num_workers=2  # 2个进程加载
    )

    # 3. 获取一个批次的数据
    for data, mask in dataloader:
        print(f"数据形状: {data.shape}")  # [4, C, H, W]
        print(f"掩码形状: {mask.shape}")  # [4, 2, H, W]
        print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")

        # 这里可以开始你的训练代码
        # model(data) 等等...
        break  # 只显示第一批数据

    # 4. 或者只获取单个样本（不通过DataLoader）
    sample_data, sample_mask = dataset[0]
    print(f"\n单个样本数据形状: {sample_data.shape}")
    print(f"单个样本掩码形状: {sample_mask.shape}")