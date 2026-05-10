"""晶圆数据集

实际目录结构（2026-04备份）：
├── BGA 12x4/         ← 产品文件夹（正常样本）
├── BGA S5E 16x7/     ← 产品文件夹（正常样本）
├── Defect/           ← 确认的缺陷样本 label=1
├── ESSD 12x4/        ← 产品文件夹（正常样本）
├── ESSD 12x5/        ← 产品文件夹（正常样本）
├── Good Unit/        ← 确认的良品样本 label=0
├── INAND 16x5/       ← 产品文件夹（正常样本）
├── INAND 19x5/       ← 产品文件夹（正常样本）
├── MicroSD 20x4/     ← 产品文件夹（正常样本）
├── Overkill/         ← 误杀（良品被判为缺陷）→ 正常样本 label=0
├── SDSIP 22x3/       ← 产品文件夹（正常样本）
├── UBGA 12x5/        ← 产品文件夹（正常样本）
└── Underkill/        ← 漏杀（缺陷被判为良品）→ 缺陷样本 label=1
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import random


# ============================================================
# 标签映射：子目录名 → 标签
# ============================================================
SUBDIR_LABELS = {
    'Defect': 1,        # 确认的缺陷
    'Underkill': 1,     # 漏杀：缺陷
    'Good Unit': 0,     # 确认的良品
    'Overkill': 0,      # 误杀：实际是良品
}

# 产品文件夹名称（不在 SUBDIR_LABELS 中的目录视为产品）
PRODUCT_FOLDERS = {
    'BGA 12x4', 'BGA S5E 16x7',
    'ESSD 12x4', 'ESSD 12x5',
    'INAND 16x5', 'INAND 19x5',
    'MicroSD 20x4',
    'SDSIP 22x3', 'UBGA 12x5',
}


# ============================================================
# 公共辅助函数
# ============================================================

def _collect_wafer_samples(root_dir):
    """
    收集晶圆数据集样本，统一返回 (normal_samples, defect_samples)
    - normal: 产品文件夹 + Good Unit + Overkill
    - defect: Defect + Underkill
    """
    root_dir = Path(root_dir)
    normal_samples = []
    defect_samples = []

    # 遍历根目录
    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue
        
        name = item.name
        if name in SUBDIR_LABELS:
            # 特殊目录：按标签分类
            label = SUBDIR_LABELS[name]
            img_paths = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
                img_paths.extend(item.glob(ext))
            for img_path in sorted(img_paths):
                if label == 0:
                    normal_samples.append(str(img_path))
                else:
                    defect_samples.append(str(img_path))
        elif name in PRODUCT_FOLDERS or (name not in ['__MACOSX', '.DS_Store'] and item.is_dir()):
            # 产品文件夹：视为正常样本
            img_paths = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
                img_paths.extend(item.rglob(ext))  # rglob 递归搜索子目录
            for img_path in sorted(img_paths):
                normal_samples.append(str(img_path))

    return normal_samples, defect_samples


def _split_samples(samples, val_ratio, split, seed=42):
    """将样本列表按 val_ratio 划分为训练/验证集"""
    if val_ratio <= 0 or split not in ['train', 'val']:
        return samples[:]

    copy = samples[:]
    rng = random.Random(seed)
    rng.shuffle(copy)
    n_val = int(len(copy) * val_ratio)
    if split == 'val':
        return copy[:n_val]
    else:
        return copy[n_val:]


# ============================================================
# 训练集（MoCo 对比学习专用）
# ============================================================

class WaferDataset(Dataset):
    """
    晶圆数据集 - 无标签预训练（兼容旧接口）
    默认只使用正常样本。可选 val_ratio 划分验证集。
    """
    def __init__(self, root_dir, transform=None, val_ratio=0.0, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.val_ratio = val_ratio
        self.split = split
        self.samples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root_dir}")

        normal_samples, _ = _collect_wafer_samples(root_dir)

        if val_ratio > 0 and split in ['train', 'val']:
            normal_split = _split_samples(normal_samples, val_ratio, split)
            self.samples = normal_split
        else:
            self.samples = normal_samples

        print(f"[INFO] 晶圆预训练数据集 ({split}): {len(self.samples)} 张正常样本 [val_ratio={val_ratio}]")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何图片！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(img)
        return img


class WaferTrainDataset(Dataset):
    """
    晶圆训练数据集 - MoCo 对比学习专用

    无监督预训练时（split='train'）：仅使用正常样本
    验证集（split='val'）：包含正常和缺陷样本
    """
    def __init__(self, root_dir, transform=None, val_ratio=0.0, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.val_ratio = val_ratio
        self.split = split
        self.samples = []
        self.labels = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root_dir}")

        normal_samples, defect_samples = _collect_wafer_samples(root_dir)

        # 划分
        if val_ratio > 0 and split in ['train', 'val']:
            normal_split = _split_samples(normal_samples, val_ratio, split)
            defect_split = _split_samples(defect_samples, val_ratio, split)
        else:
            normal_split = normal_samples[:]
            defect_split = defect_samples[:]

        # split='train': 仅使用正常样本（无监督对比学习）
        # split='val': 正常 + 缺陷（用于验证）
        if split == 'train':
            self.samples = normal_split
            self.labels = [0] * len(normal_split)
        else:
            self.samples = normal_split + defect_split
            self.labels = [0] * len(normal_split) + [1] * len(defect_split)

        n_norm = sum(1 for l in self.labels if l == 0)
        n_def  = sum(1 for l in self.labels if l == 1)
        print(f"[INFO] 晶圆训练数据集 ({split}): {len(self.samples)} 张"
              f" (正常: {n_norm}, 缺陷: {n_def}) [val_ratio={val_ratio}]")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何训练图片！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回两个增强视图 (img_q, img_k)，用于 MoCo 对比学习"""
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(img)  # SemiconductorTransform 返回 (img_q, img_k)
        return img


# ============================================================
# 评估数据集
# ============================================================

class WaferEvalDataset(Dataset):
    """
    晶圆评估数据集 - 支持从正常/缺陷样本各自划分验证集
    """
    def __init__(self, root_dir, transform=None, val_ratio=0.0, split='test'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.val_ratio = val_ratio
        self.split = split
        self.samples = []
        self.labels = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root_dir}")

        normal_samples, defect_samples = _collect_wafer_samples(root_dir)

        # 划分
        if val_ratio > 0 and split in ['train', 'val']:
            normal_split = _split_samples(normal_samples, val_ratio, split)
            defect_split = _split_samples(defect_samples, val_ratio, split)
        else:
            normal_split = normal_samples[:]
            defect_split = defect_samples[:]

        for s in normal_split:
            self.samples.append(s)
            self.labels.append(0)
        for s in defect_split:
            self.samples.append(s)
            self.labels.append(1)

        n_norm = sum(1 for l in self.labels if l == 0)
        n_def  = sum(1 for l in self.labels if l == 1)
        print(f"[INFO] 晶圆评估数据集 ({split}): {len(self.samples)} 张"
              f" (正常: {n_norm}, 缺陷: {n_def}) [val_ratio={val_ratio}]")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何评估图片！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, img_path
