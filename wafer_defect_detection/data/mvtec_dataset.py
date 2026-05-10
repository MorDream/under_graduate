"""MVTec AD数据集"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """
    MVTec AD数据集 - 工业异常检测标准数据集
    下载: https://www.mvtec.com/company/research/datasets/mvtec-ad
    """
    def __init__(self, root_dir, category, transform=None, phase='train'):
        self.root_dir = Path(root_dir) / category
        self.transform = transform
        self.phase = phase
        self.samples = []
        self.labels = []
        
        if phase == 'train':
            # 训练集只有正常样本
            good_dir = self.root_dir / 'train' / 'good'
            if good_dir.exists():
                for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
                    for f in good_dir.glob(ext):
                        self.samples.append(str(f))
                        self.labels.append(0)
        else:
            # 测试集包含正常和异常
            test_dir = self.root_dir / 'test'
            if test_dir.exists():
                for defect_type in test_dir.iterdir():
                    if not defect_type.is_dir():
                        continue
                    label = 0 if defect_type.name == 'good' else 1
                    for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
                        for f in defect_type.glob(ext):
                            self.samples.append(str(f))
                            self.labels.append(label)
        
        print(f"[INFO] MVTec {category} {phase}: {len(self.samples)} 张 (正常:{self.labels.count(0)}, 异常:{self.labels.count(1)})")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何图片！请检查数据路径")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        # SemiconductorTransform 已经返回 (img_q, img_k) tuple
        if self.transform:
            result = self.transform(img)
            # 如果 transform 返回 tuple，直接返回
            if isinstance(result, tuple):
                return result
            # 否则生成两个视图
            img_q = result
            img_k = self.transform(img)
            return img_q, img_k
        else:
            return img, img


class MVTecTrainDataset(Dataset):
    """
    MVTec AD数据集 - 训练版本（返回两个视图用于MoCo）
    """
    def __init__(self, root_dir, category, transform=None):
        self.root_dir = Path(root_dir) / category
        self.transform = transform
        self.samples = []
        
        # 训练集只有正常样本
        good_dir = self.root_dir / 'train' / 'good'
        if good_dir.exists():
            for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
                for f in good_dir.glob(ext):
                    self.samples.append(str(f))
        
        print(f"[INFO] MVTec {category} train: {len(self.samples)} 张正常样本")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何图片！请检查数据路径")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        # SemiconductorTransform 已经返回 (img_q, img_k) tuple
        if self.transform:
            result = self.transform(img)
            # 如果 transform 返回 tuple，直接返回
            if isinstance(result, tuple):
                return result
            # 否则生成两个视图
            img_q = result
            img_k = self.transform(img)
            return img_q, img_k
        else:
            return img, img


class MVTecEvalDataset(Dataset):
    """
    MVTec AD数据集 - 评估版本（返回 (img, label, path) 用于评估）
    """
    def __init__(self, root_dir, category, transform=None, phase='test'):
        self.root_dir = Path(root_dir) / category
        self.transform = transform
        self.phase = phase
        self.samples = []
        self.labels = []
        
        if phase == 'train':
            # 训练集只有正常样本
            good_dir = self.root_dir / 'train' / 'good'
            if good_dir.exists():
                for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
                    for f in good_dir.glob(ext):
                        self.samples.append(str(f))
                        self.labels.append(0)
        else:
            # 测试集包含正常和异常
            test_dir = self.root_dir / 'test'
            if test_dir.exists():
                for defect_type in test_dir.iterdir():
                    if not defect_type.is_dir():
                        continue
                    label = 0 if defect_type.name == 'good' else 1
                    for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
                        for f in defect_type.glob(ext):
                            self.samples.append(str(f))
                            self.labels.append(label)
        
        print(f"[INFO] MVTec {category} {phase}: {len(self.samples)} 张 (正常:{self.labels.count(0)}, 异常:{self.labels.count(1)})")
        if len(self.samples) == 0:
            print(f"[WARNING] 在 {self.root_dir} 中没有找到任何图片！请检查数据路径")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path


def get_mvtec_categories():
    """获取MVTec AD所有类别"""
    return [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
