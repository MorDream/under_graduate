"""
快速测试脚本 - 验证baseline各模块是否正常工作
"""

import os
import sys
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入baseline模块
from baseline.train import (
    get_device, SemiconductorTransform, EvalTransform,
    WaferDataset, WaferEvalDataset, ViTEncoder, MoCoV2,
    AnomalyDetector
)


def quick_test():
    print("=" * 60)
    print("Baseline 快速测试")
    print("=" * 60)
    
    # 1. 设备测试
    print("\n[1] 设备检测...")
    device = get_device()
    print(f"    设备: {device}")
    
    # 2. 数据增强测试
    print("\n[2] 数据增强测试...")
    transform = SemiconductorTransform(img_size=224)
    from PIL import Image
    import numpy as np
    # 创建测试图片
    test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    q, k = transform(test_img)
    print(f"    Query shape: {q.shape}")
    print(f"    Key shape: {k.shape}")
    
    # 3. 数据集加载测试
    print("\n[3] 数据集加载测试...")
    data_root = project_root / "data" / "数据集" / "数据集"
    if data_root.exists():
        eval_transform = EvalTransform(img_size=224)
        eval_dataset = WaferEvalDataset(data_root, transform=eval_transform)
        if len(eval_dataset) > 0:
            img, label, path = eval_dataset[0]
            print(f"    样本数量: {len(eval_dataset)}")
            print(f"    第一个样本: shape={img.shape}, label={label}")
        else:
            print("    [WARNING] 评估数据集为空")
    else:
        print(f"    [WARNING] 数据目录不存在: {data_root}")
    
    # 4. ViT编码器测试
    print("\n[4] ViT编码器测试...")
    encoder = ViTEncoder(img_size=224, embed_dim=384, depth=4, num_heads=6)
    test_input = torch.randn(2, 3, 224, 224)
    output = encoder(test_input)
    print(f"    输入shape: {test_input.shape}")
    print(f"    输出shape: {output.shape}")
    print(f"    CLS token shape: {output[:, 0].shape}")
    
    # 5. MoCo框架测试
    print("\n[5] MoCo框架测试...")
    moco = MoCoV2(embed_dim=384, queue_size=256, img_size=224)
    img_q = torch.randn(4, 3, 224, 224)
    img_k = torch.randn(4, 3, 224, 224)
    logits, labels, feat = moco(img_q, img_k)
    print(f"    Logits shape: {logits.shape}")
    print(f"    Labels shape: {labels.shape}")
    print(f"    Feature shape: {feat.shape}")
    
    # 6. 模型参数统计
    print("\n[6] 模型参数统计...")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"    ViT总参数: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")
    
    total_params = sum(p.numel() for p in moco.parameters())
    trainable_params = sum(p.numel() for p in moco.parameters() if p.requires_grad)
    print(f"    MoCo总参数: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")
    
    # 7. 简单训练测试（1个batch）
    print("\n[7] 简单训练测试...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(moco.parameters(), lr=0.01)
    
    moco.train()
    optimizer.zero_grad()
    logits, labels, feat = moco(img_q, img_k)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"    Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("所有测试通过! Baseline可以正常运行。")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    quick_test()