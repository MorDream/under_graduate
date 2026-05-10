"""数据增强变换"""
import random
import numpy as np
from PIL import Image
from torchvision import transforms


class CutPaste:
    """
    CutPaste数据增强 - 模拟合成异常
    参考: CutPaste: Self-Supervised Learning for Anomaly Detection (CVPR 2021)
    """
    def __init__(self, img_size=224, patch_size=64, max_patches=3, mode='uniform'):
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.mode = mode  # 'uniform', 'scar', '3way'
    
    def __call__(self, img):
        """应用CutPaste增强"""
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img.copy()
        
        h, w = img_array.shape[:2]
        
        # 根据模式选择patch大小
        if self.mode == 'scar':
            # 细长划痕
            patch_h = self.patch_size // 4
            patch_w = self.patch_size * 2
        else:
            patch_h = patch_w = self.patch_size
        
        num_patches = random.randint(1, self.max_patches)
        
        for _ in range(num_patches):
            if h <= patch_h or w <= patch_w:
                continue
            
            # 随机选择源位置
            y1 = random.randint(0, h - patch_h)
            x1 = random.randint(0, w - patch_w)
            
            # 随机选择目标位置
            y2 = random.randint(0, h - patch_h)
            x2 = random.randint(0, w - patch_w)
            
            # 执行CutPaste
            patch = img_array[y1:y1+patch_h, x1:x1+patch_w].copy()
            img_array[y2:y2+patch_h, x2:x2+patch_w] = patch
        
        return Image.fromarray(img_array)


class SemiconductorTransform:
    """
    半导体专用数据增强 - 改进版
    保持原有架构，新增CutPaste选项
    """
    def __init__(self, img_size=224, use_cutpaste=True, cutpaste_prob=0.3):
        self.img_size = img_size
        self.use_cutpaste = use_cutpaste
        self.cutpaste_prob = cutpaste_prob
        
        # CutPaste增强器
        self.cutpaste = CutPaste(img_size=img_size)
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        
        # 光度变换
        self.photo_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
        ])
        
        # Tensor变换
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        # Query分支
        img_q = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
        
        # 应用CutPaste
        if self.use_cutpaste and random.random() < self.cutpaste_prob:
            img_q = self.cutpaste(img_q)
        
        img_q = self.base_transform(img_q)
        img_q = self.photo_transform(img_q)
        img_q = self.tensor_transform(img_q)
        img_q = self.normalize(img_q)
        
        # Key分支（不使用CutPaste）
        img_k = self.base_transform(img)
        img_k = self.photo_transform(img_k)
        img_k = self.tensor_transform(img_k)
        img_k = self.normalize(img_k)
        
        return img_q, img_k


class EvalTransform:
    """评估用数据变换"""
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)
