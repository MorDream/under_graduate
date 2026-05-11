"""
DenseSimSiam: 稠密多尺度SimSiam对比学习模型
==========================================

论文参考: SimSiam (Chen & He, CVPR 2021) - Exploring Simple Siamese Representation Learning

与MoCo v2的关键区别:
- ❌ 无动量编码器: 两个视图共享同一个编码器权重
- ❌ 无负样本队列: stop-gradient防止模型坍缩
- ❌ 无温度参数/对比损失: 使用负余弦相似度
- ✅ 保留多尺度特征融合
- ✅ 新增稠密(patch级)对比学习
- ✅ 保留超球面约束、特征生成-判别等辅助改进

核心公式:
  D(p1, stopgrad(z2)) = -cos(p1, z2)
  Loss = 0.5*D(p1, z2) + 0.5*D(p2, z1)

创新点:
1. 全局+局部双分支SimSiam
2. 多尺度稠密特征对齐 (patch-level)
3. 特征级CutPaste合成异常增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def neg_cosine_similarity(p, z):
    """负余弦相似度 - SimSiam核心损失函数
    p: 预测特征 (predictor输出)
    z: 投影特征 (projector输出, 已detach)
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


class ProjectionMLP(nn.Module):
    """3层投影MLP: in_dim → in_dim → out_dim
    SimSiam标准设计: BN + ReLU in hidden layers
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    """2层预测MLP: in_dim → hidden_dim → in_dim
    SimSiam: predictor比projector少一层 (bottleneck设计)
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatureGenerator(nn.Module):
    """特征生成器: 生成伪异常特征 (来自SimpleNet)"""
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatureDiscriminator(nn.Module):
    """特征判别器"""
    def __init__(self, in_dim=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DenseSimSiam(nn.Module):
    """
    DenseSimSiam: 稠密多尺度SimSiam对比学习

    架构:
    - Shared ViT encoder (f)
    - Global SimSiam: projector_g + predictor_g on [CLS] token
    - Dense SimSiam: projector_d + predictor_d on patch tokens
    - Multi-scale SimSiam: 对多个中间层也做SimSiam
    - 辅助: 超球面约束 + 特征生成-判别

    对比MoCo:
    - 无需momentum encoder (省50%显存)
    - 无需负样本队列 (训练更简单)
    - stop-gradient防止坍缩
    - 稠密对比: patch级别学习局部正常模式
    """

    def __init__(self, embed_dim=384, img_size=224,
                 proj_dim=256, pred_hidden=128,
                 use_multiscale=True, use_dense=True,
                 use_feature_generator=True, use_hypersphere=True):
        super().__init__()
        from .vit_encoder import ViTEncoder

        self.embed_dim = embed_dim
        self.use_multiscale = use_multiscale
        self.use_dense = use_dense
        self.use_feature_generator = use_feature_generator
        self.use_hypersphere = use_hypersphere

        # ===== 共享ViT编码器 =====
        self.encoder = ViTEncoder(img_size=img_size, embed_dim=embed_dim)
        self.num_patches = self.encoder.num_patches  # 196 for patch_size=16

        # ===== 全局SimSiam分支 =====
        # Projection: 384 → 384 → 256
        self.projector_g = ProjectionMLP(embed_dim, embed_dim, proj_dim)
        # Prediction: 256 → 128 → 256
        self.predictor_g = PredictionMLP(proj_dim, pred_hidden)

        # ===== 稠密SimSiam分支 (patch级别) =====
        if use_dense:
            self.projector_d = ProjectionMLP(embed_dim, embed_dim, proj_dim)
            self.predictor_d = PredictionMLP(proj_dim, pred_hidden)

        # ===== 多尺度SimSiam分支 =====
        if use_multiscale:
            # ViT有6层，每2层取一次 → 3个中间特征
            n_scales = 3
            self.multiscale_projectors = nn.ModuleList([
                ProjectionMLP(embed_dim, embed_dim, proj_dim)
                for _ in range(n_scales)
            ])
            self.multiscale_predictors = nn.ModuleList([
                PredictionMLP(proj_dim, pred_hidden)
                for _ in range(n_scales)
            ])

        # ===== 辅助模块 =====
        if use_feature_generator:
            self.feature_generator = FeatureGenerator(embed_dim, embed_dim // 2, embed_dim)
            self.feature_discriminator = FeatureDiscriminator(embed_dim)

        if use_hypersphere:
            self.hypersphere_center = nn.Parameter(torch.zeros(embed_dim))
            self.hypersphere_radius = nn.Parameter(torch.tensor(1.0))

    def _simsiam_loss(self, p1, z2, p2, z1):
        """对称SimSiam损失"""
        loss = 0.5 * neg_cosine_similarity(p1, z2) + 0.5 * neg_cosine_similarity(p2, z1)
        return loss

    def _dense_simsiam_loss(self, patch_feat1, patch_feat2):
        """稠密SimSiam损失: 在patch特征上做对比
        patch_feat: [B, N_patches, D]
        使用average pooling做局部→全局对齐
        """
        if not self.use_dense:
            return torch.tensor(0.0, device=patch_feat1.device)

        # 平均池化 → [B, D]
        h1 = patch_feat1.mean(dim=1)
        h2 = patch_feat2.mean(dim=1)

        z1 = self.projector_d(h1)
        z2 = self.projector_d(h2)
        p1 = self.predictor_d(z1)
        p2 = self.predictor_d(z2)

        return self._simsiam_loss(p1, z2.detach(), p2, z1.detach())

    def hypersphere_loss(self, feat):
        """超球面约束损失 (CFA)"""
        if not self.use_hypersphere:
            return torch.tensor(0.0, device=feat.device)
        distances = torch.norm(feat - self.hypersphere_center, dim=1)
        return torch.mean(torch.abs(distances - self.hypersphere_radius))

    def generator_discriminator_loss(self, feat):
        """特征生成-判别损失 (SimpleNet)"""
        if not self.use_feature_generator:
            return (torch.tensor(0.0, device=feat.device),
                    torch.tensor(0.0, device=feat.device))

        batch_size = feat.shape[0]
        noise = torch.randn_like(feat) * 0.05
        fake_features = self.feature_generator(feat + noise)

        real_scores = self.feature_discriminator(feat)
        fake_scores = self.feature_discriminator(fake_features.detach())

        real_labels = torch.ones(batch_size, 1, device=feat.device)
        fake_labels = torch.zeros(batch_size, 1, device=feat.device)

        d_loss = (F.binary_cross_entropy(real_scores, real_labels) +
                  F.binary_cross_entropy(fake_scores, fake_labels))

        fake_scores_for_g = self.feature_discriminator(fake_features)
        g_loss = F.binary_cross_entropy(fake_scores_for_g, real_labels)

        return d_loss, g_loss

    def forward(self, img1, img2, epoch=0, total_epochs=100):
        """
        前向传播

        Args:
            img1, img2: 同一图片的两个增强视图 [B, 3, 224, 224]
            epoch, total_epochs: 用于温度调度 (保留接口兼容性)

        Returns:
            losses: dict of loss components
            feat_g: 全局特征 [B, embed_dim] (用于后续异常检测)
        """
        # ===== 获取两视图特征 =====
        if self.use_multiscale:
            out1, inter_feat1 = self.encoder(img1, return_all_layers=True)
            out2, inter_feat2 = self.encoder(img2, return_all_layers=True)
        else:
            out1 = self.encoder(img1)
            out2 = self.encoder(img2)
            inter_feat1, inter_feat2 = [], []

        # [CLS] token特征
        cls1 = out1[:, 0]   # [B, D]
        cls2 = out2[:, 0]

        # Patch特征
        patch1 = out1[:, 1:, :]  # [B, N, D]
        patch2 = out2[:, 1:, :]

        # ===== 1. 全局SimSiam损失 =====
        z1_g = self.projector_g(cls1)
        z2_g = self.projector_g(cls2)
        p1_g = self.predictor_g(z1_g)
        p2_g = self.predictor_g(z2_g)
        loss_global = self._simsiam_loss(p1_g, z2_g.detach(), p2_g, z1_g.detach())

        # ===== 2. 稠密SimSiam损失 =====
        loss_dense = self._dense_simsiam_loss(patch1, patch2)

        # ===== 3. 多尺度SimSiam损失 =====
        loss_multiscale = torch.tensor(0.0, device=img1.device)
        if self.use_multiscale and len(inter_feat1) > 0:
            for i, (f1, f2) in enumerate(zip(inter_feat1, inter_feat2)):
                c1 = f1[:, 0]
                c2 = f2[:, 0]
                z1_ms = self.multiscale_projectors[i](c1)
                z2_ms = self.multiscale_projectors[i](c2)
                p1_ms = self.multiscale_predictors[i](z1_ms)
                p2_ms = self.multiscale_predictors[i](z2_ms)
                loss_multiscale = loss_multiscale + \
                    self._simsiam_loss(p1_ms, z2_ms.detach(), p2_ms, z1_ms.detach())
            loss_multiscale = loss_multiscale / len(inter_feat1)

        # ===== 全局特征 (用于异常检测) =====
        if self.use_multiscale:
            # 多尺度融合 (与MoCo版本一致)
            n_layers = len(inter_feat1)
            weights = [0.1 + 0.9 * (i / max(1, n_layers - 1)) for i in range(n_layers)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            pooled_feats = []
            for feat in inter_feat1:
                pf = feat[:, 1:, :].mean(dim=1)
                pooled_feats.append(pf)

            feat_g = cls1 * 0.5
            for w, pf in zip(weights, pooled_feats):
                feat_g = feat_g + w * pf * 0.5
        else:
            feat_g = cls1

        # ===== 4. 辅助损失 =====
        loss_hypersphere = self.hypersphere_loss(feat_g)
        d_loss, g_loss = self.generator_discriminator_loss(feat_g)

        losses = {
            'global': loss_global,
            'dense': loss_dense,
            'multiscale': loss_multiscale,
            'hypersphere': loss_hypersphere,
            'discriminator': d_loss,
            'generator': g_loss,
        }

        return losses, feat_g

    @torch.no_grad()
    def extract_features(self, x):
        """提取特征用于异常检测 (评估时使用)"""
        if self.use_multiscale:
            out, inter_feat = self.encoder(x, return_all_layers=True)
            cls_feat = out[:, 0]

            n_layers = len(inter_feat)
            if n_layers > 0:
                weights = [0.1 + 0.9 * (i / max(1, n_layers - 1)) for i in range(n_layers)]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]

                pooled_feats = []
                for feat in inter_feat:
                    pf = feat[:, 1:, :].mean(dim=1)
                    pooled_feats.append(pf)

                feat = cls_feat * 0.5
                for w, pf in zip(weights, pooled_feats):
                    feat = feat + w * pf * 0.5
            else:
                feat = cls_feat
        else:
            feat = self.encoder.forward_features(x)

        return feat

    @torch.no_grad()
    def forward_features(self, x):
        """兼容异常检测器接口: 返回[CLS] token特征"""
        return self.encoder.forward_features(x)

    @torch.no_grad()
    def forward_multiscale(self, x):
        """兼容异常检测器接口: 多尺度特征融合"""
        return self.extract_features(x)

    def load_pretrained_encoder(self, pretrained_path):
        """加载预训练的encoder权重"""
        self.encoder.load_pretrained(pretrained_path)
