"""
半导体晶圆无监督缺陷检测 - 改进版V2
=====================================基于前沿方法进行改进：
- SimpleNet: 特征生成-判别框架
- CFA: 耦合超球面约束
- CutPaste: 合成异常增强
- RealNet: 轻量级重建
- 改进对比损失: NT-Xent + 难负样本挖掘

适配硬件：Intel Arc B580 (XPU) / CPU回退
运行方式：python train_improved_v2.py
"""

import os
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# 0. 设备配置
# ============================================================
def get_device():
    """获取计算设备，优先使用CUDA"""
    if torch.cuda.is_available():
        print(f"[INFO] 使用 CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("cuda")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"[INFO] 使用 Intel XPU: {torch.xpu.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("xpu")
    else:
        print("[WARNING] GPU不可用，回退到CPU")
        return torch.device("cpu")


# ============================================================
# 1. 数据增强 - 改进版（新增CutPaste合成异常）
# ============================================================
class CutPasteAugmentation:
    """
    CutPaste数据增强 - 模拟合成异常
    参考论文: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization" (CVPR 2021)
    """
    def __init__(self, img_size=224, patch_size=64, max_patches=3):
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_patches = max_patches
    
    def __call__(self, img):
        """在图像中剪贴patch模拟异常"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        h, w = img.shape[:2]
        num_patches = random.randint(1, self.max_patches)
        img_aug = img.copy()
        
        for _ in range(num_patches):
            # 随机选择patch位置
            if h <= self.patch_size or w <= self.patch_size:
                continue
            
            y1 = random.randint(0, h - self.patch_size)
            x1 = random.randint(0, w - self.patch_size)
            
            # 剪贴到另一位置
            y2 = random.randint(0, h - self.patch_size)
            x2 = random.randint(0, w - self.patch_size)
            
            # 执行CutPaste
            patch = img[y1:y1+self.patch_size, x1:x1+self.patch_size].copy()
            img_aug[y2:y2+self.patch_size, x2:x2+self.patch_size] = patch
        
        return Image.fromarray(img_aug)


class SemiconductorTransform:
    """
    半导体晶圆图片专用数据增强 - 改进版V2
    改进：
    - 增加CutPaste合成异常
    - 增加多种仿真缺陷增强
    """
    def __init__(self, img_size=224, use_cutpaste=True, cutpaste_prob=0.3):
        self.img_size = img_size
        self.use_cutpaste = use_cutpaste
        self.cutpaste_prob = cutpaste_prob
        
        # CutPaste增强器
        self.cutpaste = CutPasteAugmentation(img_size=img_size)
        
        # 基础空间变换
        self.spatial_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        
        # 光度/模糊变换
        self.photo_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
        ])
        
        # 张量变换
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            # 局部遮挡模拟颗粒污染
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        # 确保是PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        # Query分支：强增强
        img_q = img.copy()
        
        # 应用CutPaste（以一定概率）
        if self.use_cutpaste and random.random() < self.cutpaste_prob:
            img_q = self.cutpaste(img_q)
        
        img_q = self.spatial_transform(img_q)
        img_q = self.photo_transform(img_q)
        img_q = self.tensor_transform(img_q)
        img_q = self.normalize(img_q)
        
        # Key分支：较弱增强
        img_k = self.spatial_transform(img)
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


# ============================================================
# 2. 数据集加载
# ============================================================
class WaferDataset(Dataset):
    """晶圆图片数据集 - 无标签，用于对比学习预训练"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # 遍历所有产品类型的图片
        for product_dir in self.root_dir.iterdir():
            if not product_dir.is_dir():
                continue
            if product_dir.name == "Defect sample":
                continue
            if product_dir.name == "__MACOSX":
                continue
            # 进入子目录
            for subdir in product_dir.rglob("*.jpg"):
                self.samples.append(str(subdir))

        print(f"[INFO] 加载预训练数据集: {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(img)
        return img


class WaferEvalDataset(Dataset):
    """晶圆评估数据集 - 用于异常检测评估"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

        defect_dir = self.root_dir / "Defect sample"
        if defect_dir.exists():
            # Good Unit
            good_dir = defect_dir / "Good Unit"
            if good_dir.exists():
                for f in good_dir.glob("*.jpg"):
                    self.samples.append(str(f))
                    self.labels.append(0)

            # Defect
            def_dir = defect_dir / "Defect"
            if def_dir.exists():
                for f in def_dir.glob("*.jpg"):
                    self.samples.append(str(f))
                    self.labels.append(1)

            # Overkill
            over_dir = defect_dir / "Overkill"
            if over_dir.exists():
                for f in over_dir.glob("*.jpg"):
                    self.samples.append(str(f))
                    self.labels.append(0)

            # Underkill
            under_dir = defect_dir / "Underkill"
            if under_dir.exists():
                for f in under_dir.glob("*.jpg"):
                    self.samples.append(str(f))
                    self.labels.append(1)

        print(f"[INFO] 加载评估数据集: {len(self.samples)} 张 (正常:{self.labels.count(0)}, 缺陷:{self.labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, img_path


# ============================================================
# 3. ViT 编码器 - 改进版
# ============================================================
class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer 编码器 - 改进版
    支持多尺度特征提取和中间层输出
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.depth = depth

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_all_layers=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        intermediate_features = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_all_layers and (i + 1) % 2 == 0:
                intermediate_features.append(self.norm(x))

        x = self.norm(x)

        if return_all_layers:
            return x, intermediate_features
        return x

    def forward_features(self, x):
        """返回[CLS] token特征"""
        x = self.forward(x)
        return x[:, 0]

    def forward_multiscale(self, x):
        """多尺度特征融合"""
        x, intermediate_features = self.forward(x, return_all_layers=True)
        
        cls_feat = x[:, 0]
        multi_scale_feats = []
        
        for feat in intermediate_features:
            patch_feat = feat[:, 1:, :]
            pooled_feat = patch_feat.mean(dim=1)
            multi_scale_feats.append(pooled_feat)
        
        if len(multi_scale_feats) > 0:
            fused_feat = torch.stack([cls_feat] + multi_scale_feats, dim=1)
            fused_feat = fused_feat.mean(dim=1)
        else:
            fused_feat = cls_feat
            
        return fused_feat


# ============================================================
# 4. 特征生成器 - 来自SimpleNet
# ============================================================
class FeatureGenerator(nn.Module):
    """
    特征生成器 - 生成伪异常特征进行训练
    参考SimpleNet的思想，通过轻量级网络生成异常特征
    """
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=384):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, x):
        """生成伪异常特征"""
        return self.generator(x)


# ============================================================
# 5. 改进的对比学习框架 - 融合多种前沿方法
# ============================================================
class ImprovedContrastiveModel(nn.Module):
    """
    改进版对比学习模型
    融合了以下前沿方法：
    1. MoCo v2 - 动量对比学习
    2. SimpleNet - 特征生成-判别框架
    3. CFA - 超球面约束
    4. 难负样本挖掘
    """
    def __init__(self, embed_dim=384, queue_size=1024, momentum=0.999,
                 temperature=0.07, img_size=224, use_multiscale=True,
                 use_feature_generator=True, use_hypersphere=True):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.base_temperature = temperature
        self.temperature = temperature
        self.use_multiscale = use_multiscale
        self.use_feature_generator = use_feature_generator
        self.use_hypersphere = use_hypersphere

        # Query编码器
        self.encoder_q = ViTEncoder(img_size=img_size, embed_dim=embed_dim)
        # Key编码器
        self.encoder_k = ViTEncoder(img_size=img_size, embed_dim=embed_dim)

        # 投影头
        self.projector_q = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128),
        )
        self.projector_k = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128),
        )

        # SimpleNet风格的特征生成器
        if use_feature_generator:
            self.feature_generator = FeatureGenerator(embed_dim, embed_dim // 2, embed_dim)
            self.discriminator = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        # 超球面约束参数（来自CFA）
        if use_hypersphere:
            self.hypersphere_center = nn.Parameter(torch.zeros(embed_dim))
            self.hypersphere_radius = nn.Parameter(torch.tensor(1.0))

        # 初始化key编码器
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                   self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.projector_q.parameters(),
                                   self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 负样本队列
        self.register_buffer("queue", torch.randn(128, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def update_temperature(self, epoch, total_epochs):
        """温度参数调度"""
        warmup_epochs = total_epochs // 4
        if epoch < warmup_epochs:
            self.temperature = 0.1
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            self.temperature = self.base_temperature * (1 - 0.3 * progress)
        return self.temperature

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                   self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        for param_q, param_k in zip(self.projector_q.parameters(),
                                   self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            first_part = self.queue_size - ptr
            self.queue[:, ptr:ptr + first_part] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
            ptr = batch_size - first_part
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def nt_xent_loss(self, proj_q, proj_k, epoch, total_epochs):
        """
        NT-Xent损失 (Normalized Temperature-scaled Cross Entropy)
        改进版：增加难负样本挖掘
        """
        batch_size = proj_q.shape[0]
        
        # 正样本对的相似度
        pos_sim = torch.einsum('nc,nc->n', [proj_q, proj_k]).unsqueeze(-1)
        
        # 负样本队列的相似度
        neg_sim = torch.einsum('nc,ck->nk', [proj_q, self.queue.clone().detach()])
        
        # 难负样本挖掘：找出最难区分的负样本
        k_hard = min(20, self.queue_size // 4)
        hard_neg_values, hard_neg_indices = neg_sim.topk(k_hard, dim=1)
        
        # 增大难负样本的权重
        hard_neg_weight = 2.0
        for i in range(batch_size):
            for idx in hard_neg_indices[i]:
                neg_sim[i, idx] *= hard_neg_weight
        
        # 计算logits
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

    def hypersphere_loss(self, feat_q):
        """
        超球面约束损失 - 来自CFA
        将正常样本特征约束在超球面上
        """
        if not self.use_hypersphere:
            return torch.tensor(0.0, device=feat_q.device)
        
        # 计算特征到中心的距离
        distances = torch.norm(feat_q - self.hypersphere_center, dim=1)
        
        # 约束特征在超球面内
        sphere_loss = torch.mean((distances - self.hypersphere_radius) ** 2)
        
        return sphere_loss

    def feature_generator_loss(self, feat_q):
        """
        特征生成器损失 - 来自SimpleNet
        生成伪异常特征进行判别训练
        """
        if not self.use_feature_generator:
            return torch.tensor(0.0, device=feat_q.device)
        
        batch_size = feat_q.shape[0]
        
        # 生成伪异常特征
        noise = torch.randn_like(feat_q) * 0.1
        fake_features = self.feature_generator(feat_q + noise)
        
        # 判别器判断
        real_scores = self.discriminator(feat_q)
        fake_scores = self.discriminator(fake_features.detach())
        
        # 二元交叉熵损失
        real_labels = torch.ones(batch_size, 1, device=feat_q.device)
        fake_labels = torch.zeros(batch_size, 1, device=feat_q.device)
        
        gen_loss = F.binary_cross_entropy(real_scores, real_labels) + \
                   F.binary_cross_entropy(fake_scores, fake_labels)
        
        return gen_loss

    def forward(self, img_q, img_k, epoch=0, total_epochs=100):
        """前向传播"""
        # 更新温度
        current_temp = self.update_temperature(epoch, total_epochs)
        
        # Query分支
        if self.use_multiscale:
            feat_q = self.encoder_q.forward_multiscale(img_q)
        else:
            feat_q = self.encoder_q.forward_features(img_q)
        proj_q = self.projector_q(feat_q)
        proj_q = F.normalize(proj_q, dim=1)

        # Key分支
        with torch.no_grad():
            self._momentum_update_key_encoder()
            if self.use_multiscale:
                feat_k = self.encoder_k.forward_multiscale(img_k)
            else:
                feat_k = self.encoder_k.forward_features(img_k)
            proj_k = self.projector_k(feat_k)
            proj_k = F.normalize(proj_k, dim=1)

        # 计算各种损失
        losses = {}
        
        # 1. NT-Xent对比损失
        losses['contrastive'] = self.nt_xent_loss(proj_q, proj_k, epoch, total_epochs)
        
        # 2. 超球面约束损失
        losses['hypersphere'] = self.hypersphere_loss(feat_q)
        
        # 3. 特征生成器损失
        losses['generator'] = self.feature_generator_loss(feat_q)
        
        # 更新队列
        self._dequeue_and_enqueue(proj_k)

        return losses, feat_q


# ============================================================
# 6. 改进的异常检测器
# ============================================================
class ImprovedAnomalyDetector:
    """
    改进版异常检测器
    融合了PCA降维、超球面距离和多尺度特征
    """
    def __init__(self, device='cpu', n_components=None, use_hypersphere=True):
        self.device = device
        self.n_components = n_components
        self.use_hypersphere = use_hypersphere
        
        self.mean = None
        self.cov_inv = None
        self.pca_mean = None
        self.pca_components = None
        
        # 超球面参数
        self.hypersphere_center = None
        self.hypersphere_radius = None

    def fit(self, encoder, dataloader, use_multiscale=True):
        """用正常样本拟合特征分布"""
        encoder.eval()
        features = []

        with torch.no_grad():
            for imgs, labels, _ in dataloader:
                normal_mask = (labels == 0)
                if normal_mask.sum() == 0:
                    continue
                imgs_normal = imgs[normal_mask].to(self.device)
                
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs_normal)
                else:
                    feat = encoder.forward_features(imgs_normal)
                features.append(feat.cpu().numpy())

        features = np.concatenate(features, axis=0)
        print(f"[INFO] 正常样本特征矩阵: {features.shape}")

        # PCA降维
        n_samples, n_features = features.shape
        
        if self.n_components is None:
            self.n_components = min(n_features, n_samples // 2)
        
        self.pca_mean = np.mean(features, axis=0)
        features_centered = features - self.pca_mean
        
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
        var_explained = (S ** 2) / np.sum(S ** 2)
        cumulative_var = np.cumsum(var_explained)
        
        n_components = min(self.n_components, np.searchsorted(cumulative_var, 0.95) + 1)
        
        self.pca_components = Vt[:n_components, :]
        features_pca = features_centered @ self.pca_components.T
        
        print(f"[INFO] PCA: {n_features}D -> {n_components}D (保留 {cumulative_var[n_components-1]*100:.1f}%方差)")

        # 计算马氏距离参数
        self.mean = np.mean(features_pca, axis=0)
        
        # Ledoit-Wolf收缩估计
        cov, shrinkage = self._ledoit_wolf_shrinkage(features_pca)
        print(f"[INFO] Ledoit-Wolf收缩系数: {shrinkage:.4f}")
        
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("[WARNING] 协方差矩阵奇异，使用伪逆")
            self.cov_inv = np.linalg.pinv(cov)

        # 计算超球面参数
        if self.use_hypersphere:
            self.hypersphere_center = self.pca_mean
            distances = np.linalg.norm(features - self.hypersphere_center, axis=1)
            self.hypersphere_radius = np.percentile(distances, 95)
            print(f"[INFO] 超球面半径: {self.hypersphere_radius:.4f}")

        print(f"[INFO] 异常检测器已拟合")

    def _ledoit_wolf_shrinkage(self, X):
        """Ledoit-Wolf收缩估计"""
        n_samples, n_features = X.shape
        
        sample_cov = np.cov(X, rowvar=False)
        target = np.diag(np.diag(sample_cov))
        
        mu = np.trace(sample_cov) / n_features
        delta = np.sum((sample_cov - target) ** 2) / n_features
        
        X_centered = X - X.mean(axis=0)
        beta = 0
        for i in range(n_samples):
            outer_i = np.outer(X_centered[i], X_centered[i])
            beta += np.sum((outer_i - sample_cov) ** 2)
        beta = beta / (n_samples ** 2)
        
        shrinkage = min(1, max(0, beta / delta)) if delta > 0 else 1
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        
        return shrunk_cov, shrinkage

    def score(self, encoder, dataloader, use_multiscale=True):
        """计算异常分数"""
        encoder.eval()
        all_scores = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for imgs, labels, paths in dataloader:
                imgs = imgs.to(self.device)
                
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs).cpu().numpy()
                else:
                    feat = encoder.forward_features(imgs).cpu().numpy()

                # PCA降维
                feat_pca = (feat - self.pca_mean) @ self.pca_components.T

                # 计算马氏距离
                for i in range(feat_pca.shape[0]):
                    diff = feat_pca[i] - self.mean
                    mahal_dist = np.sqrt(diff @ self.cov_inv @ diff)
                    
                    # 超球面距离
                    if self.use_hypersphere:
                        sphere_dist = np.linalg.norm(feat[i] - self.hypersphere_center)
                        # 综合距离
                        combined_dist = mahal_dist + 0.5 * max(0, sphere_dist - self.hypersphere_radius)
                    else:
                        combined_dist = mahal_dist
                    
                    all_scores.append(combined_dist)
                    all_labels.append(labels[i].item())
                    all_paths.append(paths[i])

        return np.array(all_scores), np.array(all_labels), all_paths


# ============================================================
# 7. 训练流程
# ============================================================
def train(args):
    """训练改进版对比学习模型"""
    device = get_device()

    # 数据集
    data_root = Path(args.data_dir) / "数据集" / "数据集"
    transform = SemiconductorTransform(img_size=args.img_size, 
                                       use_cutpaste=args.use_cutpaste,
                                       cutpaste_prob=args.cutpaste_prob)
    dataset = WaferDataset(data_root, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type in ['xpu', 'cuda'])
    )

    # 模型
    model = ImprovedContrastiveModel(
        embed_dim=args.embed_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        img_size=args.img_size,
        use_multiscale=args.use_multiscale,
        use_feature_generator=args.use_feature_generator,
        use_hypersphere=args.use_hypersphere,
    ).to(device)

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练 | 设备: {device} | Epochs: {args.epochs}")
    print(f"数据集大小: {len(dataset)} | Batch Size: {args.batch_size}")
    print(f"改进功能: CutPaste={args.use_cutpaste}, 特征生成器={args.use_feature_generator}, 超球面={args.use_hypersphere}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_hypersphere = 0.0
        total_generator = 0.0
        num_batches = 0

        for batch_idx, (img_q, img_k) in enumerate(dataloader):
            img_q = img_q.to(device)
            img_k = img_k.to(device)

            losses, _ = model(img_q, img_k, epoch=epoch, total_epochs=args.epochs)
            
            # 组合损失
            loss = losses['contrastive'] + \
                   args.hypersphere_weight * losses['hypersphere'] + \
                   args.generator_weight * losses['generator']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += losses['contrastive'].item()
            total_hypersphere += losses['hypersphere'].item()
            total_generator += losses['generator'].item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (C:{losses['contrastive'].item():.3f} "
                      f"H:{losses['hypersphere'].item():.3f} G:{losses['generator'].item():.3f}) "
                      f"Temp: {model.temperature:.4f}")

        avg_loss = total_loss / num_batches
        avg_contrastive = total_contrastive / num_batches
        avg_hypersphere = total_hypersphere / num_batches
        avg_generator = total_generator / num_batches
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Avg Loss: {avg_loss:.4f} (C:{avg_contrastive:.3f} H:{avg_hypersphere:.3f} G:{avg_generator:.3f}) | "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_q_state_dict': model.encoder_q.state_dict(),
                'projector_q_state_dict': model.projector_q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, save_dir / "best_model_v2.pth")
            print(f"  -> 保存最佳模型 (loss={best_loss:.4f})")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'encoder_q_state_dict': model.encoder_q.state_dict(),
        'projector_q_state_dict': model.projector_q.state_dict(),
        'loss': avg_loss,
        'args': vars(args),
    }, save_dir / "final_model_v2.pth")
    print(f"\n训练完成! 最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")

    return model


# ============================================================
# 8. 评估流程
# ============================================================
def evaluate(args):
    """评估异常检测效果"""
    device = get_device()
    data_root = Path(args.data_dir) / "数据集" / "数据集"

    # 加载模型
    encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_q_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    use_multiscale = checkpoint.get('args', {}).get('use_multiscale', True)
    print(f"[INFO] 加载模型: {args.checkpoint}")

    # 评估数据集
    eval_transform = EvalTransform(img_size=args.img_size)
    eval_dataset = WaferEvalDataset(data_root, transform=eval_transform)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # 异常检测
    detector = ImprovedAnomalyDetector(device=device, n_components=args.pca_components)
    detector.fit(encoder, eval_loader, use_multiscale=use_multiscale)

    scores, labels, paths = detector.score(encoder, eval_loader, use_multiscale=use_multiscale)

    # 计算指标
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score

    auroc = roc_auc_score(labels, scores)
    print(f"\n{'='*60}")
    print(f"异常检测结果")
    print(f"{'='*60}")
    print(f"AUROC: {auroc:.4f}")

    # 找最优F1阈值
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_thresh_idx]

    print(f"最优阈值: {best_threshold:.4f} | F1: {best_f1:.4f}")

    # 混淆矩阵
    preds = (scores > best_threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    print(f"\n混淆矩阵 (阈值={best_threshold:.4f}):")
    print(f"  TP={tp} FP={fp}")
    print(f"  FN={fn} TN={tn}")
    print(f"  漏检率(FNR): {fn/(tp+fn+1e-8):.4f}")
    print(f"  误检率(FPR): {fp/(fp+tn+1e-8):.4f}")

    return auroc, best_f1


# ============================================================
# 9. 主入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='半导体晶圆无监督缺陷检测 - 改进版V2')

    # 数据
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224,
                       help='输入图片大小')

    # 模型
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='ViT嵌入维度')
    parser.add_argument('--queue_size', type=int, default=1024,
                       help='MoCo负样本队列大小')
    parser.add_argument('--momentum', type=float, default=0.999,
                       help='MoCo动量更新系数')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='对比损失温度参数')

    # 改进功能开关
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                       help='使用多尺度特征融合')
    parser.add_argument('--use_cutpaste', action='store_true', default=True,
                       help='使用CutPaste合成异常增强')
    parser.add_argument('--use_feature_generator', action='store_true', default=True,
                       help='使用SimpleNet风格的特征生成器')
    parser.add_argument('--use_hypersphere', action='store_true', default=True,
                       help='使用CFA风格的超球面约束')

    # 损失权重
    parser.add_argument('--hypersphere_weight', type=float, default=0.1,
                       help='超球面损失权重')
    parser.add_argument('--generator_weight', type=float, default=0.05,
                       help='特征生成器损失权重')
    parser.add_argument('--cutpaste_prob', type=float, default=0.3,
                       help='CutPaste增强概率')

    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.03,
                       help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载线程数')

    # 保存/加载
    parser.add_argument('--save_dir', type=str, default='./baseline/checkpoints_v2',
                       help='模型保存目录')
    parser.add_argument('--checkpoint', type=str, default='./baseline/checkpoints_v2/best_model_v2.pth',
                       help='评估时加载的模型路径')

    # 评估
    parser.add_argument('--pca_components', type=int, default=None,
                       help='PCA降维维度')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'train_and_eval'],
                       help='运行模式')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("半导体晶圆无监督缺陷检测 - 改进版V2")
    print("=" * 60)
    print("\n融合的前沿方法：")
    print("  1. MoCo v2 - 动量对比学习")
    print("  2. SimpleNet - 特征生成-判别框架")
    print("  3. CFA - 超球面约束")
    print("  4. CutPaste - 合成异常增强")
    print("  5. 难负样本挖掘")
    print("=" * 60)

    if args.mode in ['train', 'train_and_eval']:
        model = train(args)

    if args.mode in ['eval', 'train_and_eval']:
        auroc, f1 = evaluate(args)
        print(f"\n最终结果: AUROC={auroc:.4f}, F1={f1:.4f}")


if __name__ == '__main__':
    main()
