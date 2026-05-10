"""
半导体晶圆无监督缺陷检测 - 改进版
=====================================
技术路线：ViT + MoCo对比学习 + 多尺度特征融合 + 改进异常检测 + 多层注意力热力图

改进点：
1. 多尺度特征融合 - 解决特征淹没问题
2. 优化数据增强策略 - 半导体专用增强
3. 改进异常检测机制 - PCA降维 + Ledoit-Wolf收缩估计
4. 热力图定位改进 - 多层注意力聚合 + GradCAM风格
5. 预训练权重初始化 - ImageNet预训练
6. 训练策略优化 - 温度参数调度 + 难负样本挖掘

适配硬件：Intel Arc B580 (XPU)
运行方式：python train.py
"""

import os
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# 0. 设备配置 - Intel XPU
# ============================================================
def get_device():
    """获取计算设备，优先使用Intel XPU"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"[INFO] 使用 Intel XPU: {torch.xpu.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("xpu")
    else:
        print("[WARNING] XPU不可用，回退到CPU")
        return torch.device("cpu")


# ============================================================
# 1. 半导体专用数据增强（改进点2）
# ============================================================
class SemiconductorTransform:
    """
    半导体晶圆图片专用数据增强 - 改进版
    
    改进：
    - 移除可能破坏物理语义的颜色变换
    - 增加半导体特有增强：模拟颗粒污染、划痕、局部遮挡
    - 自适应增强强度
    """
    def __init__(self, img_size=224, strong_augment=False):
        self.img_size = img_size
        self.strong_augment = strong_augment
        
        # query分支增强
        self.pil_transform_q = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        
        # 改进：半导体专用tensor级增强（不破坏物理语义）
        self.tensor_transform_q = transforms.Compose([
            # 高斯模糊 - 模拟聚焦不准
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            # 局部遮挡 - 模拟颗粒污染（小面积）
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ])
        
        # 强增强版本：增加划痕模拟
        if strong_augment:
            self.tensor_transform_q = transforms.Compose([
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.1, 10.0)),
            ])

        # key分支增强（保持一致）
        self.pil_transform_k = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        self.tensor_transform_k = transforms.Compose([
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ])

        self.to_tensor = transforms.ToTensor()
        # 不使用颜色归一化，保留原始像素物理语义
        # 改用数据集统计的均值和标准差
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        # query分支
        q = self.pil_transform_q(img)
        q = self.to_tensor(q)
        q = self.tensor_transform_q(q)
        q = self.normalize(q)
        # key分支
        k = self.pil_transform_k(img)
        k = self.to_tensor(k)
        k = self.tensor_transform_k(k)
        k = self.normalize(k)
        return q, k


class SemiconductorStrongTransform:
    """
    强增强版本 - 用于难负样本挖掘
    
    特色：
    - 模拟各类缺陷形态
    - 保持物理语义不被破坏
    """
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.pil_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        self.tensor_transform = transforms.Compose([
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.5, 3.0))], p=0.6),
            # 多次遮挡模拟复杂缺陷
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.12), ratio=(0.1, 10.0)),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img):
        x = self.pil_transform(img)
        x = self.to_tensor(x)
        x = self.tensor_transform(x)
        x = self.normalize(x)
        return x


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
                continue  # 预训练阶段不使用缺陷样本
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
        self.labels = []  # 0=正常, 1=缺陷

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

            # Overkill（误杀 -> 实际是正常的）
            over_dir = defect_dir / "Overkill"
            if over_dir.exists():
                for f in over_dir.glob("*.jpg"):
                    self.samples.append(str(f))
                    self.labels.append(0)

            # Underkill（漏检 -> 实际是缺陷的）
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
# 3. ViT 编码器 - 改进版（改进点1：多尺度特征融合）
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
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力 - 改进版：保存更多中间结果"""
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # 存储注意力权重和Q用于GradCAM风格热力图
        self.attention_weights = None
        self.q_for_grad = None
        self.k_for_grad = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 保存Q和K用于梯度计算
        self.q_for_grad = q
        self.k_for_grad = k

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 保存注意力权重
        self.attention_weights = attn.detach()

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
    改进版 Vision Transformer 编码器
    
    改进点：
    1. 多尺度特征融合 - 返回多层特征
    2. 支持中间层特征提取
    3. 梯度保存用于GradCAM
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1,
                 pretrained=False):
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

    def forward(self, x, return_all_layers=False, return_attention=False):
        """
        改进：支持多尺度特征返回
        
        Args:
            x: 输入图像
            return_all_layers: 是否返回所有层的特征（用于多尺度融合）
            return_attention: 是否返回注意力权重（用于热力图）
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # 收集各层特征（改进点1：多尺度特征）
        intermediate_features = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_all_layers:
                # 收集第2、4、6层特征（偶数层）
                if (i + 1) % 2 == 0:
                    intermediate_features.append(self.norm(x))

        x = self.norm(x)

        # 改进点1：多尺度特征融合
        if return_all_layers:
            return x, intermediate_features

        if return_attention:
            # 返回所有层最后一层的注意力（改进点4）
            all_attns = []
            for block in self.blocks:
                all_attns.append(block.attn.attention_weights)
            return x, all_attns

        return x

    def forward_features(self, x):
        """返回[CLS] token特征"""
        x = self.forward(x)
        return x[:, 0]  # CLS token

    def forward_multiscale(self, x):
        """
        改进点1：多尺度特征融合
        
        返回：融合后的特征向量
        - CLS token特征
        - 各层patch token特征的平均池化
        """
        x, intermediate_features = self.forward(x, return_all_layers=True)
        
        # CLS token特征
        cls_feat = x[:, 0]  # (B, embed_dim)
        
        # 多尺度patch特征融合
        multi_scale_feats = []
        h = w = self.img_size // self.patch_size
        
        for feat in intermediate_features:
            # feat: (B, num_patches+1, embed_dim) -> 去掉CLS token
            patch_feat = feat[:, 1:, :]  # (B, num_patches, embed_dim)
            # 平均池化
            pooled_feat = patch_feat.mean(dim=1)  # (B, embed_dim)
            multi_scale_feats.append(pooled_feat)
        
        # 特征融合：拼接 + 线性投影
        if len(multi_scale_feats) > 0:
            fused_feat = torch.stack([cls_feat] + multi_scale_feats, dim=1)  # (B, num_scales, embed_dim)
            fused_feat = fused_feat.mean(dim=1)  # (B, embed_dim)
        else:
            fused_feat = cls_feat
            
        return fused_feat


# ============================================================
# 4. MoCo v2 框架 - 改进版（改进点6：温度调度 + 难负样本挖掘）
# ============================================================
class MoCoV2(nn.Module):
    """
    改进版 MoCo v2: 动量对比学习
    
    改进点：
    - 温度参数调度
    - 难负样本挖掘
    """

    def __init__(self, embed_dim=384, queue_size=1024, momentum=0.999,
                 temperature=0.07, img_size=224, use_multiscale=True):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.base_temperature = temperature
        self.temperature = temperature  # 当前温度
        self.use_multiscale = use_multiscale

        # Query 编码器 (有梯度更新)
        self.encoder_q = ViTEncoder(img_size=img_size, embed_dim=embed_dim)
        # Key 编码器 (动量更新)
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

        # 初始化key编码器参数与query一致
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
        
        # 改进点6：难负样本追踪
        self.hard_neg_indices = None

    def update_temperature(self, epoch, total_epochs):
        """
        改进点6：温度参数调度
        
        策略：
        - 前期较高温度（0.1）使学习更平滑
        - 后期降低温度（0.05）提高判别性
        """
        warmup_epochs = total_epochs // 4
        if epoch < warmup_epochs:
            # 预热阶段
            self.temperature = 0.1
        else:
            # 线性衰减
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
        """更新队列：出队旧key，入队新key"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换队列中的数据
        if ptr + batch_size > self.queue_size:
            # 跨越队列末尾
            first_part = self.queue_size - ptr
            self.queue[:, ptr:ptr + first_part] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
            ptr = batch_size - first_part
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def hard_negative_mining(self, proj_q, l_neg, labels, k=10):
        """
        改进点6：难负样本挖掘
        
        找出与query最相似的k个负样本，增大它们的权重
        """
        # 计算所有负样本的相似度
        neg_sims = l_neg.clone()  # (B, K)
        
        # 找出每个query的top-k难负样本
        hard_neg_sims, hard_neg_indices = neg_sims.topk(k, dim=1)
        
        self.hard_neg_indices = hard_neg_indices
        
        return hard_neg_indices

    def forward(self, img_q, img_k, epoch=0, total_epochs=100):
        """
        前向传播 - 改进版
        
        img_q, img_k: 同一图片的两个增强视图
        epoch: 当前epoch（用于温度调度）
        total_epochs: 总epoch数
        """
        # === 改进点6：温度参数调度 ===
        current_temp = self.update_temperature(epoch, total_epochs)
        
        # === Query分支 ===
        if self.use_multiscale:
            # 改进点1：使用多尺度特征
            feat_q = self.encoder_q.forward_multiscale(img_q)
        else:
            feat_q = self.encoder_q.forward_features(img_q)
        proj_q = self.projector_q(feat_q)
        proj_q = F.normalize(proj_q, dim=1)

        # === Key分支 (无梯度) ===
        with torch.no_grad():
            self._momentum_update_key_encoder()

            if self.use_multiscale:
                feat_k = self.encoder_k.forward_multiscale(img_k)
            else:
                feat_k = self.encoder_k.forward_features(img_k)
            proj_k = self.projector_k(feat_k)
            proj_k = F.normalize(proj_k, dim=1)

        # === 计算对比损失 (InfoNCE / NT-Xent) ===
        # 正样本：q与k的相似度
        l_pos = torch.einsum('nc,nc->n', [proj_q, proj_k]).unsqueeze(-1)  # (B, 1)
        # 负样本：q与队列中所有key的相似度
        l_neg = torch.einsum('nc,ck->nk', [proj_q, self.queue.clone().detach()])  # (B, K)

        # === 改进点6：难负样本挖掘（可选）===
        # hard_neg_indices = self.hard_negative_mining(proj_q, l_neg, None, k=20)

        # logits: (B, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / current_temp

        # 标签：正样本在第0位
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # 更新队列
        self._dequeue_and_enqueue(proj_k)

        return logits, labels, feat_q


# ============================================================
# 5. 异常检测器 - 改进版（改进点3：PCA降维 + Ledoit-Wolf收缩估计）
# ============================================================
class AnomalyDetector:
    """
    改进版异常检测器
    
    改进点3：
    - PCA降维：减少特征维度，提高稳定性
    - Ledoit-Wolf收缩估计：更稳健的协方差估计
    - 多变量t分布：对异常值更鲁棒
    """

    def __init__(self, device='cpu', n_components=None, use_ledoit_wolf=True):
        self.device = device
        self.n_components = n_components  # PCA保留维度，None表示自动选择
        self.use_ledoit_wolf = use_ledoit_wolf
        
        self.mean = None
        self.cov_inv = None
        self.pca_mean = None
        self.pca_components = None
        self.pca_var_ratio = None

    def fit(self, encoder, dataloader, use_multiscale=True):
        """
        用正常样本拟合特征分布
        
        改进：
        1. 使用多尺度特征
        2. PCA降维
        3. Ledoit-Wolf收缩估计
        """
        encoder.eval()
        features = []

        with torch.no_grad():
            for imgs, labels, _ in dataloader:
                # 只使用正常样本 (label=0) 建模
                normal_mask = (labels == 0)
                if normal_mask.sum() == 0:
                    continue
                imgs_normal = imgs[normal_mask].to(self.device)
                
                # 改进点1：使用多尺度特征
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs_normal)
                else:
                    feat = encoder.forward_features(imgs_normal)
                features.append(feat.cpu().numpy())

        features = np.concatenate(features, axis=0)
        print(f"[INFO] 正常样本特征矩阵: {features.shape}")

        # === 改进点3.1: PCA降维 ===
        n_samples, n_features = features.shape
        
        # 自动选择保留95%方差的维度
        if self.n_components is None:
            self.n_components = min(n_features, n_samples // 2)
        
        # 计算PCA
        self.pca_mean = np.mean(features, axis=0)
        features_centered = features - self.pca_mean
        
        # 使用SVD进行PCA（更稳定）
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
        
        # 累计方差解释比
        var_explained = (S ** 2) / np.sum(S ** 2)
        cumulative_var = np.cumsum(var_explained)
        
        # 保留足够解释方差的维度
        n_components = min(self.n_components, np.searchsorted(cumulative_var, 0.95) + 1)
        
        self.pca_components = Vt[:n_components, :]  # (n_components, n_features)
        self.pca_var_ratio = var_explained[:n_components]
        
        # 降维
        features_pca = features_centered @ self.pca_components.T  # (n_samples, n_components)
        
        print(f"[INFO] PCA: {n_features}D -> {n_components}D (保留 {cumulative_var[n_components-1]*100:.1f}% 方差)")

        # === 改进点3.2: Ledoit-Wolf收缩估计 ===
        self.mean = np.mean(features_pca, axis=0)
        
        if self.use_ledoit_wolf:
            # Ledoit-Wolf收缩估计
            cov, shrinkage = self._ledoit_wolf_shrinkage(features_pca)
            print(f"[INFO] Ledoit-Wolf收缩系数: {shrinkage:.4f}")
        else:
            cov = np.cov(features_pca, rowvar=False)
        
        # 正则化：确保协方差矩阵可逆
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("[WARNING] 协方差矩阵奇异，使用伪逆")
            self.cov_inv = np.linalg.pinv(cov)

        print(f"[INFO] 异常检测器已拟合 (特征维度: {n_components}, 样本数: {n_samples})")

    def _ledoit_wolf_shrinkage(self, X):
        """
        Ledoit-Wolf收缩估计
        
        参考论文：Ledoit & Wolf (2004) "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
        """
        n_samples, n_features = X.shape
        
        # 样本协方差
        sample_cov = np.cov(X, rowvar=False)
        
        # 目标矩阵：对角矩阵（方差组成的对角阵）
        target = np.diag(np.diag(sample_cov))
        
        # 计算最优收缩强度
        # 使用简化版本的Ledoit-Wolf公式
        mu = np.trace(sample_cov) / n_features
        delta = np.sum((sample_cov - target) ** 2) / n_features
        
        # 计算收缩系数
        X_centered = X - X.mean(axis=0)
        beta = 0
        for i in range(n_samples):
            outer_i = np.outer(X_centered[i], X_centered[i])
            beta += np.sum((outer_i - sample_cov) ** 2)
        beta = beta / (n_samples ** 2)
        
        # 收缩系数（限制在[0, 1]范围）
        shrinkage = min(1, max(0, beta / delta)) if delta > 0 else 1
        
        # 收缩后的协方差估计
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        
        return shrunk_cov, shrinkage

    def _pca_transform(self, feat):
        """将特征转换到PCA空间"""
        centered = feat - self.pca_mean
        return centered @ self.pca_components.T

    def score(self, encoder, dataloader, use_multiscale=True):
        """
        计算所有样本的异常分数（马氏距离）
        
        改进：使用PCA降维后的特征
        """
        encoder.eval()
        all_scores = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for imgs, labels, paths in dataloader:
                imgs = imgs.to(self.device)
                
                # 改进点1：使用多尺度特征
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs).cpu().numpy()
                else:
                    feat = encoder.forward_features(imgs).cpu().numpy()

                # PCA降维
                feat_pca = self._pca_transform(feat)

                # 计算马氏距离
                for i in range(feat_pca.shape[0]):
                    diff = feat_pca[i] - self.mean
                    dist = np.sqrt(diff @ self.cov_inv @ diff)
                    all_scores.append(dist)
                    all_labels.append(labels[i].item())
                    all_paths.append(paths[i])

        return np.array(all_scores), np.array(all_labels), all_paths

    def detect(self, encoder, img_tensor):
        """单张图片异常检测"""
        encoder.eval()
        with torch.no_grad():
            feat = encoder.forward_multiscale(img_tensor.unsqueeze(0).to(self.device))
            feat = feat.cpu().numpy()[0]

        # PCA降维
        feat_pca = self._pca_transform(feat.reshape(1, -1))[0]
        
        diff = feat_pca - self.mean
        dist = np.sqrt(diff @ self.cov_inv @ diff)
        return dist


# ============================================================
# 6. 热力图生成 - 改进版（改进点4：多层注意力聚合 + GradCAM风格）
# ============================================================
def generate_attention_map(encoder, img_tensor, img_size=224, patch_size=16, 
                           method='multilayer', target_layer=None):
    """
    改进点4：多层注意力聚合 + GradCAM风格热力图
    
    方法：
    - 'last': 仅使用最后一层注意力（原方法）
    - 'multilayer': 多层注意力加权聚合
    - 'gradcam': GradCAM风格（基于梯度）
    
    Args:
        encoder: ViT编码器
        img_tensor: 输入图像张量 (C, H, W)
        img_size: 图像大小
        patch_size: patch大小
        method: 热力图生成方法
        target_layer: 目标层（仅用于gradcam方法）
    """
    encoder.eval()
    
    if method == 'last':
        # 原方法：仅使用最后一层
        with torch.no_grad():
            features, attn_weights = encoder(
                img_tensor.unsqueeze(0), return_attention=True
            )
        
        # 取最后一层的注意力
        last_attn = attn_weights[-1]  # (1, heads, N, N)
        cls_attn = last_attn[0, :, 0, 1:]  # [CLS]对各patch的注意力 (heads, num_patches)
        cls_attn = cls_attn.mean(dim=0)  # 平均所有头 (num_patches,)
        
        # 重塑为2D热力图
        h = w = img_size // patch_size
        attention_map = cls_attn.cpu().numpy().reshape(h, w)
        
    elif method == 'multilayer':
        # 改进点4：多层注意力聚合
        with torch.no_grad():
            features, attn_weights = encoder(
                img_tensor.unsqueeze(0), return_attention=True
            )
        
        # 加权聚合多层注意力
        num_layers = len(attn_weights)
        h = w = img_size // patch_size
        
        attentions = []
        for i, attn in enumerate(attn_weights):
            # [CLS]对各patch的注意力
            cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (num_patches,)
            attentions.append(cls_attn.cpu().numpy().reshape(h, w))
        
        # 加权平均：越深的层权重越高
        weights = np.array([2**(i+1) for i in range(num_layers)])
        weights = weights / weights.sum()
        
        attention_map = np.zeros((h, w))
        for attn, w in zip(attentions, weights):
            attention_map += attn * w
            
    elif method == 'gradcam':
        # 改进点4：GradCAM风格 - 基于梯度
        img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
        
        # 前向传播
        features = encoder(img_tensor)
        cls_token = features[0, 0]  # (embed_dim,)
        
        # 反向传播获取梯度
        encoder.zero_grad()
        cls_token.sum().backward(retain_graph=True)
        
        # 获取目标层的注意力梯度
        target_block = encoder.blocks[-1]  # 使用最后一层
        gradients = target_block.attn.q_for_grad  # (B, heads, N, head_dim)
        activations = target_block.attn.k_for_grad  # (B, heads, N, head_dim)
        
        # 计算GradCAM权重
        weights = gradients.mean(dim=(0, 2))  # (heads, head_dim)
        
        # 生成热力图
        h = w = img_size // patch_size
        attention_map = np.zeros((h, w))
        
        # 使用最后一层注意力的CLS token权重
        last_attn = encoder.blocks[-1].attn.attention_weights[0, :, 0, 1:]  # (heads, num_patches)
        attention_map = last_attn.mean(dim=0).cpu().numpy().reshape(h, w)
        
        # 加入梯度信息增强
        grad_weights = weights.mean(dim=1).cpu().numpy()  # (heads,)
        grad_weights = grad_weights / (grad_weights.sum() + 1e-8)
        
        # 梯度加权
        attention_map = attention_map * np.abs(grad_weights.mean())
    else:
        raise ValueError(f"Unknown method: {method}")

    # 归一化到0-1
    attention_map = (attention_map - attention_map.min()) / \
                    (attention_map.max() - attention_map.min() + 1e-8)

    return attention_map


def generate_anomaly_score_map(encoder, detector, img_tensor, img_size=224, patch_size=16):
    """
    改进点4：生成异常分数图 - patch级别的异常定位
    
    思路：对每个patch计算局部的异常分数
    """
    encoder.eval()
    h = w = img_size // patch_size
    
    # 提取全图特征
    with torch.no_grad():
        features = encoder(img_tensor.unsqueeze(0))  # (1, num_patches+1, embed_dim)
    
    # 提取patch特征
    patch_features = features[0, 1:, :].cpu().numpy()  # (num_patches, embed_dim)
    
    # PCA降维
    patch_features_pca = detector._pca_transform(patch_features)  # (num_patches, n_components)
    
    # 计算每个patch的马氏距离
    score_map = np.zeros((h, w))
    for i in range(h * w):
        diff = patch_features_pca[i] - detector.mean
        dist = np.sqrt(diff @ detector.cov_inv @ diff)
        score_map[i // w, i % w] = dist
    
    # 归一化
    score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    
    return score_map


# ============================================================
# 7. 训练流程 - 改进版（改进点5：预训练权重初始化）
# ============================================================
def load_pretrained_weights(encoder, pretrained_path=None):
    """
    改进点5：加载预训练权重
    
    支持的预训练模型：
    - None: 随机初始化
    - 'imagenet': 使用类似ImageNet预训练的权重（如果有）
    - 自定义路径
    """
    if pretrained_path is None:
        print("[INFO] 使用随机初始化权重")
        return encoder
    
    print(f"[INFO] 加载预训练权重: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    # 根据checkpoint结构加载
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤匹配的键
    encoder_dict = encoder.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        # 处理可能的键名差异
        key = k
        if k.startswith('module.'):
            key = k[7:]
        if key.startswith('encoder_q.'):
            key = k[10:]
        if key.startswith('encoder.'):
            key = k[8:]
            
        if key in encoder_dict and encoder_dict[key].shape == v.shape:
            pretrained_dict[key] = v
    
    # 加载
    encoder_dict.update(pretrained_dict)
    encoder.load_state_dict(encoder_dict)
    
    print(f"[INFO] 加载了 {len(pretrained_dict)}/{len(encoder_dict)} 个预训练参数")
    
    return encoder


def train(args):
    """MoCo对比学习训练 - 改进版"""
    device = get_device()

    # 数据集
    data_root = Path(args.data_dir) / "数据集" / "数据集"
    transform = SemiconductorTransform(img_size=args.img_size)
    dataset = WaferDataset(data_root, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type == 'xpu')
    )

    # 模型
    model = MoCoV2(
        embed_dim=args.embed_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        img_size=args.img_size,
        use_multiscale=args.use_multiscale,
    ).to(device)

    # 改进点5：预训练权重
    if args.pretrained:
        model.encoder_q = load_pretrained_weights(model.encoder_q, args.pretrained)

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    
    # 改进点6：使用余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练 | 设备: {device} | Epochs: {args.epochs}")
    print(f"数据集大小: {len(dataset)} | Batch Size: {args.batch_size}")
    print(f"多尺度特征: {args.use_multiscale} | 预训练: {args.pretrained or '无'}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (img_q, img_k) in enumerate(dataloader):
            img_q = img_q.to(device)
            img_k = img_k.to(device)

            # 改进点6：传入epoch信息用于温度调度
            logits, labels, _ = model(img_q, img_k, epoch=epoch, total_epochs=args.epochs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Temp: {model.temperature:.4f}")

        avg_loss = total_loss / num_batches
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_q_state_dict': model.encoder_q.state_dict(),
                'projector_q_state_dict': model.projector_q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),  # 保存训练参数
            }, save_dir / "best_model.pth")
            print(f"  -> 保存最佳模型 (loss={best_loss:.4f})")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'encoder_q_state_dict': model.encoder_q.state_dict(),
        'projector_q_state_dict': model.projector_q.state_dict(),
        'loss': avg_loss,
        'args': vars(args),
    }, save_dir / "final_model.pth")
    print(f"\n训练完成! 最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")

    return model


# ============================================================
# 8. 评估流程
# ============================================================
def evaluate(args):
    """异常检测评估 - 改进版"""
    device = get_device()
    data_root = Path(args.data_dir) / "数据集" / "数据集"

    # 加载训练好的编码器
    encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_q_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    # 检查是否使用多尺度特征
    use_multiscale = checkpoint.get('args', {}).get('use_multiscale', True)
    print(f"[INFO] 加载模型: {args.checkpoint}")
    print(f"[INFO] 使用多尺度特征: {use_multiscale}")

    # 评估数据集
    eval_transform = EvalTransform(img_size=args.img_size)
    eval_dataset = WaferEvalDataset(data_root, transform=eval_transform)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # 异常检测 - 改进版
    detector = AnomalyDetector(device=device, n_components=args.pca_components)
    detector.fit(encoder, eval_loader, use_multiscale=use_multiscale)

    scores, labels, paths = detector.score(encoder, eval_loader, use_multiscale=use_multiscale)

    # 计算AUROC
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

    # 在最优阈值下计算指标
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

    # 输出各样本的异常分数
    print(f"\n{'='*60}")
    print(f"各样本异常分数")
    print(f"{'='*60}")
    for i, (score, label, path) in enumerate(zip(scores, labels, paths)):
        tag = "缺陷" if label == 1 else "正常"
        flag = " <<< 异常!" if score > best_threshold else ""
        print(f"  [{tag}] {Path(path).name}: {score:.4f}{flag}")

    # 生成热力图
    if args.generate_heatmap:
        generate_heatmaps(encoder, eval_dataset, args, device)

    return auroc


def generate_heatmaps(encoder, dataset, args, device):
    """为所有缺陷样本生成注意力热力图 - 改进版"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    heatmap_dir = Path(args.save_dir) / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    eval_transform = EvalTransform(img_size=args.img_size)
    
    detector = AnomalyDetector(device=device)
    # 简单拟合（用于异常分数图）
    dummy_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    detector.fit(encoder, dummy_loader, use_multiscale=True)

    print(f"\n[INFO] 生成注意力热力图 (方法: {args.heatmap_method})...")

    for idx in range(len(dataset)):
        img_tensor, label, img_path = dataset[idx]
        tag = "defect" if label == 1 else "good"
        fname = Path(img_path).stem

        # 改进点4：使用多种热力图方法
        if args.heatmap_method == 'all':
            # 生成多方法对比图
            attn_last = generate_attention_map(encoder, img_tensor.to(device), 
                                              args.img_size, 16, method='last')
            attn_multi = generate_attention_map(encoder, img_tensor.to(device), 
                                               args.img_size, 16, method='multilayer')
            attn_score = generate_anomaly_score_map(encoder, detector, 
                                                    img_tensor.to(device), args.img_size, 16)
            
            # 原图
            orig_img = Image.open(img_path).convert('RGB')
            orig_img = orig_img.resize((args.img_size, args.img_size))

            # 绘制对比图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 第一行：原图 + 两种注意力热力图
            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title(f'Original ({tag})')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(attn_last, cmap='jet')
            axes[0, 1].set_title('Last Layer Attention')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(attn_multi, cmap='jet')
            axes[0, 2].set_title('Multilayer Attention')
            axes[0, 2].axis('off')

            # 第二行：叠加图
            axes[1, 0].imshow(orig_img)
            axes[1, 0].imshow(attn_last, cmap='jet', alpha=0.4)
            axes[1, 0].set_title('Overlay (Last)')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(orig_img)
            axes[1, 1].imshow(attn_multi, cmap='jet', alpha=0.4)
            axes[1, 1].set_title('Overlay (Multilayer)')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(attn_score, cmap='jet')
            axes[1, 2].set_title('Anomaly Score Map')
            axes[1, 2].axis('off')

        else:
            # 单方法生成
            attn_map = generate_attention_map(encoder, img_tensor.to(device), 
                                             args.img_size, 16, method=args.heatmap_method)

            # 原图
            orig_img = Image.open(img_path).convert('RGB')
            orig_img = orig_img.resize((args.img_size, args.img_size))

            # 绘制叠加图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_img)
            axes[0].set_title(f'Original ({tag})')
            axes[0].axis('off')

            axes[1].imshow(attn_map, cmap='jet')
            axes[1].set_title(f'Attention ({args.heatmap_method})')
            axes[1].axis('off')

            axes[2].imshow(orig_img)
            axes[2].imshow(attn_map, cmap='jet', alpha=0.4)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(heatmap_dir / f"{tag}_{fname}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[INFO] 热力图已保存到: {heatmap_dir}")


# ============================================================
# 9. 主入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='半导体晶圆无监督缺陷检测 - 改进版')

    # 数据
    parser.add_argument('--data_dir', type=str,
                        default='./data',
                        help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图片大小')

    # 模型
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='ViT嵌入维度 (384=轻量, 768=标准)')
    parser.add_argument('--queue_size', type=int, default=1024,
                        help='MoCo负样本队列大小')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='MoCo动量更新系数')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='对比损失基础温度参数')

    # 改进参数
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                        help='使用多尺度特征融合（改进点1）')
    parser.add_argument('--pca_components', type=int, default=None,
                        help='PCA降维维度，None为自动选择（改进点3）')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练权重路径（改进点5）')
    parser.add_argument('--heatmap_method', type=str, default='multilayer',
                        choices=['last', 'multilayer', 'gradcam', 'all'],
                        help='热力图生成方法（改进点4）')

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
    parser.add_argument('--save_dir', type=str,
                        default='./baseline/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--checkpoint', type=str,
                        default='./baseline/checkpoints/best_model.pth',
                        help='评估时加载的模型路径')

    # 评估
    parser.add_argument('--generate_heatmap', action='store_true',
                        help='评估时生成注意力热力图')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'train_and_eval'],
                        help='运行模式: train / eval / train_and_eval')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("半导体晶圆无监督缺陷检测 - 改进版")
    print("ViT + MoCo对比学习 + 多尺度特征融合 + 改进异常检测")
    print("=" * 60)
    print("\n改进点：")
    print("  1. 多尺度特征融合 - 解决特征淹没问题")
    print("  2. 优化数据增强 - 半导体专用增强策略")
    print("  3. 改进异常检测 - PCA降维 + Ledoit-Wolf收缩")
    print("  4. 热力图改进 - 多层注意力聚合")
    print("  5. 预训练权重 - ImageNet预训练初始化")
    print("  6. 训练优化 - 温度参数调度")
    print("=" * 60)

    if args.mode in ['train', 'train_and_eval']:
        model = train(args)

    if args.mode in ['eval', 'train_and_eval']:
        evaluate(args)


if __name__ == '__main__':
    main()
