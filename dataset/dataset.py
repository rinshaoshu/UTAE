import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import json
from transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotate90,
    Normalize,
    Compose,
    _flip_h,
    _flip_v,
    _rot90k
)

class SEN12Dataset(Dataset):
    def __init__(self, txt_path, data_dir=None ,norm_dir=None, augment=False):
        self.txt_path = txt_path
        self.data_dir = str(data_dir)
        self.augment = augment
        self.norm_dir = norm_dir
        # 配置：文件夹 -> 波段列表
        self.bands_config = [
            {'dir': 's2', 'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                                    'B08', 'B8A', 'B11', 'B12', 'DEM']},
            {'dir': 'dsc', 'bands': ['DVH', 'DVV']},
            {'dir': 'asc', 'bands': ['AVV', 'AVH']}
        ]

        # 读取文件名（不过滤）
        with open(txt_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]

        self.all_bands = []
        for config in self.bands_config:
            self.all_bands.extend(config['bands'])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        all_data = []

        for config in self.bands_config:
            file_path = os.path.join(self.data_dir, config['dir'], self.file_list[idx])
            with xr.open_dataset(file_path) as ds:
                ds = ds.sortby("time")
                data = np.stack([ds[band].values for band in config['bands']], axis=0)
                data = torch.from_numpy(data).float().permute(1, 0, 2, 3)
                all_data.append(data)

        data = torch.cat(all_data, dim=1)


        # mask 从第一个文件（s2）依次读取
        if os.path.isfile(os.path.join(self.data_dir, 's2', self.file_list[idx])):
            first_path = os.path.join(self.data_dir, 's2', self.file_list[idx])
        elif os.path.isfile(os.path.join(self.data_dir, 'asc', self.file_list[idx])):
            first_path = os.path.join(self.data_dir, 'asc', self.file_list[idx])
        elif os.path.isfile(os.path.join(self.data_dir, 'dsc', self.file_list[idx])):
            first_path = os.path.join(self.data_dir, 'dsc', self.file_list[idx])
        with xr.open_dataset(first_path) as ds:
            mask_data = ds['MASK'].isel(time=0).values
            mask_data = (mask_data > 0).astype(np.int64)
            mask = torch.from_numpy(mask_data).long()

        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # one-hot: 通道0=无效区域, 通道1=有效区域 → (2, H, W)
        mask_one_hot = torch.zeros(2, mask.shape[0], mask.shape[1], dtype=torch.float32)
        mask_one_hot[0, :, :] = (mask == 0).float()
        mask_one_hot[1, :, :] = (mask == 1).float()

        if self.augment:
            data, mask_one_hot = self._augment(data, mask_one_hot)

        return data, mask_one_hot

    def _augment(self, data, mask):
        transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(self.norm_dir)
        ])
        data, mask = transform(data, mask)
        return data, mask

# ==================== 使用示例 ====================
from torch.utils.data import DataLoader

# ==================== 使用示例 ====================
