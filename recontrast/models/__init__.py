"""
ReContrast Models

提供ResNet和ViT两种编码器架构的ReContrast模型
"""

# ResNet版本 (原始)
from .recontrast import ReContrast, ReDistill

# ViT版本 (新增)
try:
    from .recontrast_vit import (
        ReContrastViT,
        BottleneckViT,
        DecoderViT,
        load_dinov2_encoder,
        build_recontrast_vit
    )
    HAS_VIT = True
except ImportError:
    HAS_VIT = False

__all__ = [
    'ReContrast',
    'ReDistill',
]

if HAS_VIT:
    __all__.extend([
        'ReContrastViT',
        'BottleneckViT', 
        'DecoderViT',
        'load_dinov2_encoder',
        'build_recontrast_vit'
    ])
