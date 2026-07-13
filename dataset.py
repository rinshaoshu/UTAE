import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import json



class SEN12Dataset(Dataset):
    def __init__(self, txt_path, data_dir=None, json_path=None, augment=False):
        self.txt_path = txt_path
        self.data_dir = data_dir
        self.augment = augment

        # 读取文件名
        with open(txt_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]

        # 配置：文件夹 -> 波段列表
        self.bands_config = [
            {'dir': 's2', 'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                                    'B08', 'B8A', 'B11', 'B12', 'DEM']},
            {'dir': 'dsc', 'bands': ['DVH', 'DVV']},
            {'dir': 'asc', 'bands': ['AVV', 'AVH']}
        ]

        # 收集所有波段
        self.all_bands = []
        for config in self.bands_config:
            self.all_bands.extend(config['bands'])

        # 从JSON读取统计量
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                stats = json.load(f)
            # 修改1: 直接从顶层读取 mean 和 std
            self.mean = torch.tensor([stats['mean'][band] for band in self.all_bands])
            self.std = torch.tensor([stats['std'][band] for band in self.all_bands])
            print(f"加载统计量成功，总波段数: {len(self.all_bands)}")
            print(f"波段列表: {self.all_bands}")
        else:
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        all_data = []
        mask = None
        file_exists = []
        channel_valid = []

        # 先检查所有文件是否存在
        for config in self.bands_config:
            file_path = os.path.join(self.data_dir, config['dir'], self.file_list[idx])
            if os.path.exists(file_path):
                file_exists.append(True)
            else:
                file_exists.append(False)

        all_exist = all(file_exists)

        # ========== 新增：统一的mask提取函数 ==========
        def extract_mask(ds, config):
            """从数据集中提取mask的通用函数"""
            if 'MASK' in ds:
                mask_data = ds['MASK'].isel(time=0).values
                mask_data = (mask_data > 0).astype(np.int64)
                return torch.from_numpy(mask_data).long()
            return None

        if all_exist:
            for config in self.bands_config:
                file_path = os.path.join(self.data_dir, config['dir'], self.file_list[idx])
                with xr.open_dataset(file_path) as ds:
                    ds = ds.sortby("time")

                    data = np.stack([ds[band].values for band in config['bands']], axis=0)
                    data = torch.from_numpy(data).float().permute(1, 0, 2, 3)
                    all_data.append(data)

                    for _ in config['bands']:
                        channel_valid.append(True)

                    # ========== 修改：从任何存在的文件夹提取mask ==========
                    if mask is None:  # 如果还没有mask，尝试提取
                        mask = extract_mask(ds, config)

            data = torch.cat(all_data, dim=1)
            mean = self.mean.view(1, -1, 1, 1)
            std = self.std.view(1, -1, 1, 1)
            data = (data - mean) / std

        else:
            h, w = 256, 256
            t = 1

            # 获取尺寸和mask（按优先级：s2 > dsc > asc）
            for i, config in enumerate(self.bands_config):
                if file_exists[i]:
                    file_path = os.path.join(self.data_dir, config['dir'], self.file_list[idx])
                    try:
                        with xr.open_dataset(file_path) as ds:
                            first_band = config['bands'][0]
                            if first_band in ds:
                                data_shape = ds[first_band].shape
                                if len(data_shape) == 3:
                                    t, h, w = data_shape
                                elif len(data_shape) == 2:
                                    h, w = data_shape

                            # ========== 修改：从存在的文件中提取mask ==========
                            if mask is None:
                                mask = extract_mask(ds, config)
                            break
                    except Exception as e:
                        print(f"警告: 无法读取 {file_path}: {e}")
                        continue

            if mask is None:
                mask = torch.ones((h, w), dtype=torch.long)

            total_channels = sum(len(config['bands']) for config in self.bands_config)
            data = torch.zeros((t, total_channels, h, w), dtype=torch.float32)
            channel_valid = [False] * total_channels

            for i, config in enumerate(self.bands_config):
                if file_exists[i]:
                    file_path = os.path.join(self.data_dir, config['dir'], self.file_list[idx])
                    try:
                        with xr.open_dataset(file_path) as ds:
                            ds = ds.sortby("time")
                            band_data = np.stack([ds[band].values for band in config['bands']], axis=0)
                            band_data = torch.from_numpy(band_data).float().permute(1, 0, 2, 3)

                            start_ch = sum(len(self.bands_config[j]['bands']) for j in range(i))
                            end_ch = start_ch + len(config['bands'])
                            data[:, start_ch:end_ch, :, :] = band_data

                            for ch in range(start_ch, end_ch):
                                channel_valid[ch] = True

                            # ========== 修改：从存在的文件中提取mask ==========
                            if mask is None:
                                mask = extract_mask(ds, config)
                    except Exception as e:
                        print(f"警告: 加载 {file_path} 失败: {e}")

            # 如果还是没有mask，创建全1
            if mask is None:
                mask = torch.ones((h, w), dtype=torch.long)

            mean = self.mean.view(1, -1, 1, 1)
            std = self.std.view(1, -1, 1, 1)
            data = (data - mean) / std

            for ch in range(total_channels):
                if not channel_valid[ch]:
                    data[:, ch, :, :] = 0

        if self.augment and mask is not None:
            data, mask = self._augment(data, mask)

        # mask: (h, w) → (1, h, w)，DataLoader batch 后为 (b, 1, h, w)
        mask = mask.unsqueeze(0)

        return data, mask
    def _augment(self, data, mask):
        if torch.rand(1) > 0.5:
            data = torch.flip(data, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1) > 0.5:
            data = torch.flip(data, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            data = torch.rot90(data, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return data, mask


# ==================== 使用示例 ====================
from torch.utils.data import DataLoader


def main():
    # 创建数据集
    train_dataset = SEN12Dataset(
        txt_path='test.txt',
        data_dir='./data',
        json_path='norm.json',  # 直接使用你的JSON文件
        augment=True
    )

    val_dataset = SEN12Dataset(
        txt_path='test.txt',
        data_dir='./data',
        json_path='norm.json',
        augment=False
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 测试
    print("=" * 50)
    print("训练集测试:")
    for data, mask in train_loader:
        print(f'Data: {data.shape}')  # (batch, t, c, h, w)
        print(f'Mask: {mask.shape}')  # (batch, h, w)
        print(f'Data范围: {data.min():.2f} - {data.max():.2f}')
        print(f'Mask唯一值: {torch.unique(mask)}')
        break

    print("\n" + "=" * 50)
    print("验证集测试:")
    for data, mask in val_loader:
        print(f'Data: {data.shape}')
        print(f'Mask: {mask.shape}')
        print(f'Data范围: {data.min():.2f} - {data.max():.2f}')
        break

    print("\n数据集信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"总波段数: {len(train_dataset.all_bands)}")
    print(f"波段列表: {train_dataset.all_bands}")


if __name__ == "__main__":
    main()