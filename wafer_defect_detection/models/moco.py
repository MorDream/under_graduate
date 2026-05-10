"""MoCo v2 + 改进模型

关键改进：
1. 修复难负样本挖掘的in-place修改bug（会污染队列）
2. 改进超球面损失（用L1代替L2，更鲁棒）
3. 改进特征生成器噪声幅度
4. 温度调度改为渐进式warmup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    """
    特征生成器 - 生成伪异常特征
    参考SimpleNet的思想
    """
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


class ImprovedMoCo(nn.Module):
    """
    改进版MoCo v2 - 保持主体架构，添加辅助损失
    
    主体: ViT Encoder + MoCo v2框架（保持不变）
    新增:
    - 超球面约束损失 (CFA)
    - 特征生成-判别损失 (SimpleNet)
    - 难负样本挖掘（修复版）
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

        # ===== 主体架构: ViT + MoCo =====
        from .vit_encoder import ViTEncoder
        
        # Query编码器
        self.encoder_q = ViTEncoder(img_size=img_size, embed_dim=embed_dim)
        # Key编码器（动量更新）
        self.encoder_k = ViTEncoder(img_size=img_size, embed_dim=embed_dim)

        # 投影头（两层MLP，输出维度128）
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

        # ===== 新增辅助模块 =====
        if use_feature_generator:
            self.feature_generator = FeatureGenerator(embed_dim, embed_dim // 2, embed_dim)
            self.feature_discriminator = FeatureDiscriminator(embed_dim)

        if use_hypersphere:
            self.hypersphere_center = nn.Parameter(torch.zeros(embed_dim))
            self.hypersphere_radius = nn.Parameter(torch.tensor(1.0))

    def update_temperature(self, epoch, total_epochs):
        """温度参数调度 - 渐进式warmup"""
        warmup_epochs = max(1, total_epochs // 10)  # 10% warmup
        if epoch < warmup_epochs:
            # warmup: 从0.2逐步降到base_temperature
            progress = epoch / warmup_epochs
            self.temperature = 0.2 - (0.2 - self.base_temperature) * progress
        else:
            # 训练后期缓慢降低
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            self.temperature = self.base_temperature * (1 - 0.2 * progress)
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

    def nt_xent_loss(self, proj_q, proj_k):
        """
        NT-Xent损失 + 难负样本挖掘（修复版）
        
        修复：不再in-place修改neg_sim，改用加权softmax
        """
        batch_size = proj_q.shape[0]
        
        # 正样本相似度
        pos_sim = torch.einsum('nc,nc->n', [proj_q, proj_k]).unsqueeze(-1)
        
        # 负样本相似度
        neg_sim = torch.einsum('nc,ck->nk', [proj_q, self.queue.clone().detach()])
        
        # 难负样本挖掘：给高相似度的负样本更高权重
        # 使用加权方式而非in-place修改，避免污染
        k_hard = min(20, self.queue_size // 4)
        hard_neg_values, _ = neg_sim.topk(k_hard, dim=1)
        # 难负样本平均相似度
        hard_mean = hard_neg_values.mean(dim=1, keepdim=True)
        # 对难负样本赋予更高权重（2x），其他保持1x
        hard_weight = 1.0 + torch.sigmoid(neg_sim - hard_mean)  # 范围在1-2之间
        
        weighted_neg_sim = neg_sim * hard_weight

        # 计算logits
        logits = torch.cat([pos_sim, weighted_neg_sim], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

    def hypersphere_loss(self, feat_q):
        """
        超球面约束损失 - 来自CFA
        改用L1 loss，更鲁棒
        """
        if not self.use_hypersphere:
            return torch.tensor(0.0, device=feat_q.device)
        
        distances = torch.norm(feat_q - self.hypersphere_center, dim=1)
        # L1 loss：对异常值更鲁棒
        sphere_loss = torch.mean(torch.abs(distances - self.hypersphere_radius))
        
        return sphere_loss

    def generator_discriminator_loss(self, feat_q):
        """
        特征生成-判别损失 - 来自SimpleNet
        改进：降低噪声幅度
        """
        if not self.use_feature_generator:
            return torch.tensor(0.0, device=feat_q.device), torch.tensor(0.0, device=feat_q.device)
        
        batch_size = feat_q.shape[0]
        
        # 生成伪异常特征（降低噪声幅度）
        noise = torch.randn_like(feat_q) * 0.05  # 从0.1降到0.05
        fake_features = self.feature_generator(feat_q + noise)
        
        # 判别器判断
        real_scores = self.feature_discriminator(feat_q)
        fake_scores = self.feature_discriminator(fake_features.detach())
        
        # 判别器损失
        real_labels = torch.ones(batch_size, 1, device=feat_q.device)
        fake_labels = torch.zeros(batch_size, 1, device=feat_q.device)
        
        d_loss = F.binary_cross_entropy(real_scores, real_labels) + \
                 F.binary_cross_entropy(fake_scores, fake_labels)
        
        # 生成器损失
        fake_scores_for_g = self.feature_discriminator(fake_features)
        g_loss = F.binary_cross_entropy(fake_scores_for_g, real_labels)
        
        return d_loss, g_loss

    def forward(self, img_q, img_k, epoch=0, total_epochs=100):
        """前向传播"""
        # 更新温度
        current_temp = self.update_temperature(epoch, total_epochs)
        
        # ===== 主体: MoCo v2前向传播 =====
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

        # ===== 计算各种损失 =====
        losses = {}
        
        # 1. 主体对比损失
        losses['contrastive'] = self.nt_xent_loss(proj_q, proj_k)
        
        # 2. 超球面约束损失（辅助）
        losses['hypersphere'] = self.hypersphere_loss(feat_q)
        
        # 3. 生成-判别损失（辅助）
        d_loss, g_loss = self.generator_discriminator_loss(feat_q)
        losses['discriminator'] = d_loss
        losses['generator'] = g_loss
        
        # 更新队列
        self._dequeue_and_enqueue(proj_k)

        return losses, feat_q
