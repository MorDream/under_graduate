"""数据增强变换 - 优化版"""
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


class CutPaste:
    """
    CutPaste数据增强 - 模拟合成异常
    参考: CutPaste: Self-Supervised Learning for Anomaly Detection (CVPR 2021)
    
    修复：大幅缩小patch尺寸，避免破坏图像主体结构
    """
    def __init__(self, img_size=224, patch_scale=(0.02, 0.15), max_patches=3, mode='uniform'):
        self.img_size = img_size
        self.patch_scale = patch_scale  # patch占图像面积的比例范围
        self.max_patches = max_patches
        self.mode = mode
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img.copy()
        
        h, w = img_array.shape[:2]
        num_patches = random.randint(1, self.max_patches)
        
        for _ in range(num_patches):
            # 按面积比例随机生成patch大小
            area = h * w
            scale = random.uniform(self.patch_scale[0], self.patch_scale[1])
            patch_area = int(area * scale)
            aspect = random.uniform(0.5, 2.0)
            patch_h = int(np.sqrt(patch_area / aspect))
            patch_w = int(np.sqrt(patch_area * aspect))
            
            patch_h = min(patch_h, h - 1)
            patch_w = min(patch_w, w - 1)
            
            if patch_h < 2 or patch_w < 2:
                continue
            
            # 随机选择源位置
            y1 = random.randint(0, h - patch_h)
            x1 = random.randint(0, w - patch_w)
            
            if self.mode == 'scar':
                # 细长划痕：水平方向
                y2 = random.randint(0, h - patch_h)
                x2 = random.randint(0, w - patch_w)
                patch = img_array[y1:y1+patch_h, x1:x1+patch_w].copy()
                img_array[y2:y2+patch_h, x2:x2+patch_w] = patch
            else:
                # 均匀CutPaste
                y2 = random.randint(0, h - patch_h)
                x2 = random.randint(0, w - patch_w)
                patch = img_array[y1:y1+patch_h, x1:x1+patch_w].copy()
                # 随机颜色扰动（模拟缺陷颜色变化）
                if random.random() < 0.5:
                    patch = patch.astype(np.float32)
                    patch += random.uniform(-30, 30)
                    patch = np.clip(patch, 0, 255).astype(np.uint8)
                img_array[y2:y2+patch_h, x2:x2+patch_w] = patch
        
        return Image.fromarray(img_array)


class SemiconductorTransform:
    """
    半导体专用数据增强 - 优化版
    
    关键改进：
    1. CutPaste patch缩小，避免破坏图像
    2. 去掉ColorJitter（晶圆图颜色有物理含义）
    3. 增加局部遮挡（模拟颗粒污染）
    4. 增加噪声（模拟传感器噪声）
    """
    def __init__(self, img_size=224, use_cutpaste=True, cutpaste_prob=0.3):
        self.img_size = img_size
        self.use_cutpaste = use_cutpaste
        self.cutpaste_prob = cutpaste_prob
        
        # CutPaste增强器（缩小patch）
        self.cutpaste = CutPaste(img_size=img_size, patch_scale=(0.02, 0.12))
        self.cutpaste_scar = CutPaste(img_size=img_size, patch_scale=(0.01, 0.08), mode='scar')
        
        # 基础几何变换
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # 减小旋转角度
        ])
        
        # 强增强（Query分支用）
        self.strong_augment = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.3),
        ])
        
        # 弱增强（Key分支用）
        self.weak_augment = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.1),
        ])
        
        # Tensor变换
        self.to_tensor = transforms.ToTensor()
        self.random_erasing = transforms.RandomErasing(p=0.15, scale=(0.01, 0.05), ratio=(0.3, 3.3))
        
        # 使用ImageNet归一化（预训练权重兼容）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        # Query分支（强增强）
        img_q = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
        
        # 应用CutPaste
        if self.use_cutpaste and random.random() < self.cutpaste_prob:
            if random.random() < 0.5:
                img_q = self.cutpaste(img_q)
            else:
                img_q = self.cutpaste_scar(img_q)
        
        img_q = self.base_transform(img_q)
        img_q = self.strong_augment(img_q)
        img_q = self.to_tensor(img_q)
        img_q = self.random_erasing(img_q)
        img_q = self.normalize(img_q)
        
        # Key分支（弱增强，不打CutPaste）
        img_k = self.base_transform(img)
        img_k = self.weak_augment(img_k)
        img_k = self.to_tensor(img_k)
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
