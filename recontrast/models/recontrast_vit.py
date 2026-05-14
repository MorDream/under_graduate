"""
ReContrast with ViT Encoder + DINOv2 Pretrained Weights
支持预训练权重加载，保持ViT编码器结构不变
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARNING] timm not installed. Please install: pip install timm")


class BottleneckViT(nn.Module):
    """ViT专用的Bottleneck - 处理patch tokens"""
    def __init__(self, embed_dim=384, hidden_dim=768, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity()
            ])
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: list of [B, N, C] tensors (multi-scale features)
        返回: list of processed features
        """
        processed = []
        for feat in x:
            # feat: [B, N, C] - batch, num_patches+1, embed_dim
            B, N, C = feat.shape
            # Process all tokens
            feat_flat = feat.reshape(B * N, C)
            out = self.mlp(feat_flat)
            out = out.reshape(B, N, -1)
            processed.append(out)
        return processed


class DecoderViT(nn.Module):
    """ViT专用Decoder - 重建多尺度特征"""
    def __init__(self, embed_dim=384, num_scales=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # 为每个尺度创建重建头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_scales)
        ])
        
    def forward(self, x):
        """
        x: list of [B, N, C] processed features
        返回: list of [B, N, C] reconstructed features
        """
        reconstructed = []
        for i, feat in enumerate(x):
            if i < len(self.heads):
                B, N, C = feat.shape
                feat_flat = feat.reshape(B * N, C)
                out = self.heads[i](feat_flat)
                out = out.reshape(B, N, C)
                reconstructed.append(out)
            else:
                reconstructed.append(feat)
        return reconstructed


class ReContrastViT(nn.Module):
    """
    ReContrast with ViT Encoder
    双分支结构: 训练分支 + 冻结分支 (动量更新)
    """
    def __init__(self, encoder, encoder_freeze, bottleneck, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder_freeze = encoder_freeze
        
        # 冻结encoder_freeze的所有参数
        for param in self.encoder_freeze.parameters():
            param.requires_grad = False
            
        self.bottleneck = bottleneck
        self.decoder = decoder
        
    def forward(self, x):
        """
        前向传播
        x: [B, 3, H, W] 输入图像
        返回: en (encoder features), de (decoder reconstructed features)
        """
        # 训练分支
        en_train = self.encoder(x, return_all_layers=True)
        
        # 冻结分支 (无梯度)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x, return_all_layers=True)
        
        # 合并双分支特征用于bottleneck
        # en_train和en_freeze都是 tuple (final_feat, intermediate_features)
        if isinstance(en_train, tuple):
            train_feat_list = [en_train[0]] + en_train[1]  # [final] + intermediates
        else:
            train_feat_list = [en_train]
            
        if isinstance(en_freeze, tuple):
            freeze_feat_list = [en_freeze[0]] + en_freeze[1]
        else:
            freeze_feat_list = [en_freeze]
        
        # 交错合并: [train_0, freeze_0, train_1, freeze_1, ...]
        en_merged = []
        for i, (t, f) in enumerate(zip(train_feat_list, freeze_feat_list)):
            # 确保维度匹配
            en_merged.append(torch.cat([t, f], dim=0))
        
        # Bottleneck处理
        bn_out = self.bottleneck(en_merged)
        
        # Decoder重建
        de = self.decoder(bn_out)
        
        # 分割回双分支
        de_train = [d.chunk(2, dim=0)[0] for d in de]
        de_freeze = [d.chunk(2, dim=0)[1] for d in de]
        
        # 交错返回: freeze特征 + train特征 (与原始ReContrast保持一致)
        en_out = freeze_feat_list + train_feat_list
        de_out = de_freeze + de_train
        
        return en_out, de_out
    
    def train(self, mode=True, encoder_bn_train=True):
        """设置训练模式"""
        self.training = mode
        if mode:
            # 主encoder根据设置决定
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.eval()
            # 冻结encoder始终eval
            self.encoder_freeze.eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.eval()
            self.encoder_freeze.eval()
            self.bottleneck.eval()
            self.decoder.eval()
        return self


def load_dinov2_encoder(model_name='vit_small_patch14_dinov2.lvd142m', pretrained=True, **kwargs):
    """
    加载DINOv2预训练模型作为编码器
    
    Args:
        model_name: timm模型名称
        pretrained: 是否加载预训练权重
        **kwargs: 额外的模型参数
    
    Returns:
        encoder: 配置好的ViT编码器
    """
    if not HAS_TIMM:
        raise ImportError("timm is required for loading DINOv2. Please install: pip install timm")
    
    # 创建模型（统一pretrained=False，手动处理权重加载）
    encoder = timm.create_model(
        model_name,
        pretrained=False,
        features_only=False,  # 我们需要完整的forward
        **kwargs
    )
    
    if pretrained:
        local_path = 'dinov2_vits14_pretrain.pth'
        if os.path.exists(local_path):
            # 从本地文件加载权重（解决远程服务器HF下载失败的问题）
            state_dict = torch.load(local_path, map_location='cpu', weights_only=True)
            if 'student' in state_dict:
                state_dict = state_dict['student']
            encoder.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained weights from local file: {local_path}")
        else:
            # 回退到timm内置的HF下载
            from timm.models._hub import load_state_dict_from_hf
            hf_id = timm.models.get_pretrained_cfg(model_name).hf_hub_id
            state_dict = load_state_dict_from_hf(hf_id, weights_only=True)
            encoder.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained weights from HuggingFace Hub: {hf_id}")
    
    # 包装以支持多尺度输出和原始ViTEncoder兼容的接口
    encoder.return_all_layers = lambda x: encoder.forward_features(x)
    
    # 保存原始的forward
    original_forward = encoder.forward
    
    def new_forward(x, return_all_layers=False):
        """兼容的forward函数"""
        if return_all_layers:
            # 返回多尺度特征: (final_layer_patch_tokens, [intermediate_patch_tokens])
            # DINOv2有12层，取第3, 6, 9, 12层
            layers_out = encoder.get_intermediate_layers(x, n=4, return_prefix_tokens=True)
            # layers_out: list of 4 tuples (patch_tokens, prefix_token)
            all_feats = [l[0] for l in layers_out]  # 只取patch tokens
            return (all_feats[-1], all_feats[:-1])  # (final, [layer0, layer1, layer2])
        else:
            return original_forward(x)
    
    encoder.forward = new_forward
    encoder.embed_dim = encoder.embed_dim if hasattr(encoder, 'embed_dim') else 384
    
    print(f"[INFO] Loaded DINOv2 encoder: {model_name}, embed_dim={encoder.embed_dim}")
    return encoder


def build_recontrast_vit(wafer_encoder=None, use_pretrained=True, pretrained_model='vit_small_patch14_dinov2.lvd142m'):
    """
    构建ViT版本的ReContrast模型
    
    Args:
        wafer_encoder: 现有的ViTEncoder实例 (可选)
        use_pretrained: 是否使用预训练权重
        pretrained_model: 预训练模型名称
    
    Returns:
        model: ReContrastViT模型
        encoder: 主encoder
        encoder_freeze: 冻结encoder
    """
    if wafer_encoder is not None:
        # 使用现有的ViTEncoder
        encoder = wafer_encoder
        # 创建冻结副本
        encoder_freeze = type(wafer_encoder)(
            img_size=wafer_encoder.img_size,
            patch_size=wafer_encoder.patch_size,
            in_channels=3,
            embed_dim=wafer_encoder.embed_dim,
            depth=wafer_encoder.depth,
            num_heads=wafer_encoder.blocks[0].attn.num_heads if wafer_encoder.blocks else 6,
            mlp_ratio=4.0,
            dropout=0.1
        )
        # 复制权重
        encoder_freeze.load_state_dict(wafer_encoder.state_dict())
        embed_dim = wafer_encoder.embed_dim
    else:
        # 使用DINOv2预训练模型
        encoder = load_dinov2_encoder(pretrained_model, pretrained=use_pretrained)
        encoder_freeze = load_dinov2_encoder(pretrained_model, pretrained=use_pretrained)
        embed_dim = encoder.embed_dim
    
    # 冻结encoder_freeze
    for param in encoder_freeze.parameters():
        param.requires_grad = False
    
    # 创建Bottleneck和Decoder
    bottleneck = BottleneckViT(embed_dim=embed_dim, hidden_dim=embed_dim*2, num_layers=3)
    decoder = DecoderViT(embed_dim=embed_dim, num_scales=6)
    
    # 创建ReContrast模型
    model = ReContrastViT(encoder, encoder_freeze, bottleneck, decoder)
    
    return model, encoder, encoder_freeze
