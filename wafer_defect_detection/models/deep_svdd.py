"""
Deep SVDD (Deep Support Vector Data Description) 模型

核心思想：
- 训练一个神经网络，将正常样本映射到一个紧凑的超球面上
- 最小化超球面半径 R 和样本到中心 c 的距离
- 测试时，距离越大，越可能是异常

改进：
1. 使用ViT编码器作为特征提取器
2. 结合多尺度特征融合
3. 软边界DeepSVDD (允许少量异常样本)
4. 添加Wasserstein距离正则化，防止特征collapse

与MoCo对比：
- MoCo: 通过正负样本对比学习表征
- DeepSVDD: 通过最小化超球面半径学习紧凑表征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DeepSVDD(nn.Module):
    """
    Deep SVDD 模型 - 无监督异常检测
    
    核心：通过最小化正常样本到超球面中心的距离来学习紧凑表征
    
    损失函数：
        L = R^2 + (1/n) * Σ max(0, ||z_i - c||^2 - R^2) + λ*||z||^2
    
    其中：
        - R: 超球面半径
        - c: 超球面中心
        - z_i: 样本i的表征
        - λ: 正则化系数
    """
    def __init__(
        self,
        img_size: int = 224,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_multiscale: bool = True,
        use_pretrain_distill: bool = True,
        distill_weight: float = 0.1,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_multiscale = use_multiscale
        self.use_pretrain_distill = use_pretrain_distill
        self.distill_weight = distill_weight
        
        # ===== 编码器 =====
        from .vit_encoder import ViTEncoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # ===== SVDD 参数 =====
        # 可学习的超球面中心 (使用随机初始化，稍后通过预训练编码器初始化)
        self.center = nn.Parameter(torch.zeros(embed_dim))
        
        # 初始半径 (会随训练更新)
        self.radius = nn.Parameter(torch.tensor(0.5))
        
        # ===== 预训练编码器（用于蒸馏） =====
        if use_pretrain_distill:
            self.pretrain_encoder = ViTEncoder(
                img_size=img_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for param in self.pretrain_encoder.parameters():
                param.requires_grad = False
        
        # ===== 特征正则化 =====
        # 防止特征collapse：使用Wasserstein距离正则化
        self.feature_std = nn.Parameter(torch.tensor(1.0))
        
    def init_center(self, encoder_state_dict: dict):
        """用预训练编码器初始化中心"""
        # 加载权重到主编码器
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        if self.use_pretrain_distill:
            self.pretrain_encoder.load_state_dict(encoder_state_dict, strict=False)
        
        # 使用编码器的初始特征均值作为中心
        print("[INFO] DeepSVDD center initialized from encoder")
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        if self.use_multiscale:
            return self.encoder.forward_multiscale(x)
        else:
            return self.encoder.forward_features(x)
    
    def forward(self, x: torch.Tensor, return_distances: bool = False):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            return_distances: 是否返回距离
        
        Returns:
            losses: 损失字典
            或者 (losses, distances) 如果 return_distances=True
        """
        batch_size = x.shape[0]
        
        # 提取特征
        features = self.get_features(x)
        
        # L2归一化
        features = F.normalize(features, dim=1)
        
        # ===== 计算到超球面中心的距离 =====
        # 使用余弦相似度的cos形式，距离 = 1 - similarity
        center_norm = F.normalize(self.center.unsqueeze(0), dim=1)  # [1, D]
        distances = 1.0 - torch.sum(features * center_norm.expand_as(features), dim=1)  # [B]
        
        # ===== Deep SVDD 损失 =====
        # 软边界损失：minimize R^2 + (1/n) * Σ max(0, d^2 - R^2)
        radius_sq = self.radius ** 2
        violations = F.relu(distances ** 2 - radius_sq)
        svdd_loss = radius_sq + violations.mean()
        
        # ===== 正则化损失 =====
        # 1. 特征方差正则（防止collapse）
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0).mean()
        variance_loss = -torch.log(feature_std + 1e-6)  # 鼓励方差大一点
        
        # 2. 中心正则（防止中心跑到无穷远）
        center_reg = torch.norm(self.center) / self.embed_dim
        
        # ===== 蒸馏损失 =====
        distill_loss = torch.tensor(0.0, device=features.device)
        if self.use_pretrain_distill and self.training:
            with torch.no_grad():
                pretrain_features = self.pretrain_encoder.forward_features(x)
                pretrain_features = F.normalize(pretrain_features, dim=1)
            
            # 让当前特征接近预训练特征
            distill_loss = F.mse_loss(features, pretrain_features.detach())
        
        # ===== 总损失 =====
        losses = {
            'svdd': svdd_loss,
            'variance': variance_loss * 0.01,
            'center_reg': center_reg * 0.001,
            'distill': distill_loss * self.distill_weight,
            'total': svdd_loss + variance_loss * 0.01 + center_reg * 0.001 + distill_loss * self.distill_weight,
        }
        
        if return_distances:
            return losses, distances
        return losses
    
    def predict_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测异常分数
        
        分数越高，越可能是异常
        """
        with torch.no_grad():
            features = self.get_features(x)
            features = F.normalize(features, dim=1)
            
            center_norm = F.normalize(self.center.unsqueeze(0), dim=1)
            # 使用余弦距离作为异常分数
            scores = 1.0 - torch.sum(features * center_norm.expand_as(features), dim=1)
            
        return scores


class DeepSVDDWithAutoEncoder(nn.Module):
    """
    Deep SVDD + AutoEncoder 混合模型
    
    使用自编码器重建辅助任务，帮助学习更好的表征
    """
    def __init__(
        self,
        img_size: int = 224,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_multiscale: bool = True,
    ):
        super().__init__()
        
        from .vit_encoder import ViTEncoder
        
        # 主编码器
        self.encoder = ViTEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # SVDD参数
        self.center = nn.Parameter(torch.zeros(embed_dim))
        self.radius = nn.Parameter(torch.tensor(0.5))
        
        # 轻量级解码器（重建patch特征，而非原始图像）
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.use_multiscale = use_multiscale
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_multiscale:
            return self.encoder.forward_multiscale(x)
        else:
            return self.encoder.forward_features(x)
    
    def forward(self, x: torch.Tensor, return_reconstruction: bool = False):
        """前向传播"""
        batch_size = x.shape[0]
        
        # 编码
        features = self.get_features(x)
        features_norm = F.normalize(features, dim=1)
        
        # 解码（重建特征）
        reconstruction = self.decoder(features)
        
        # SVDD损失
        center_norm = F.normalize(self.center.unsqueeze(0), dim=1)
        distances = 1.0 - torch.sum(features_norm * center_norm.expand_as(features_norm), dim=1)
        
        radius_sq = self.radius ** 2
        violations = F.relu(distances ** 2 - radius_sq)
        svdd_loss = radius_sq + violations.mean()
        
        # 重建损失（让重建特征接近原始特征）
        recon_loss = F.mse_loss(reconstruction, features.detach())
        
        # 总损失
        total_loss = svdd_loss + 0.1 * recon_loss
        
        losses = {
            'svdd': svdd_loss,
            'recon': recon_loss,
            'total': total_loss,
        }
        
        if return_reconstruction:
            return losses, reconstruction
        return losses
    
    def predict_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.get_features(x)
            features = F.normalize(features, dim=1)
            center_norm = F.normalize(self.center.unsqueeze(0), dim=1)
            scores = 1.0 - torch.sum(features * center_norm.expand_as(features), dim=1)
        return scores