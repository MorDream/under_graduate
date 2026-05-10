"""ViT编码器"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """图像分块嵌入 - 保持不变"""
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
    """多头自注意力 - 保持不变"""
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
    """Transformer块 - 保持不变"""
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
    Vision Transformer编码器 - 保持原有架构
    支持多尺度特征提取
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
        """返回CLS token特征"""
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
