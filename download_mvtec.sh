#!/bin/bash
# MVTec AD数据集下载脚本
# 由于官方需要申请，这里提供社区镜像下载方式

echo "=========================================="
echo "MVTec AD Anomaly Detection Dataset 下载"
echo "=========================================="
echo ""

# 创建数据集目录
DATASET_DIR="./mvtec_anomaly_detection"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

# 方式1: Kaggle下载 (推荐)
echo "方式1: 通过Kaggle下载 (需要安装kaggle)"
echo "请先确保:"
echo "  1. 安装kaggle: pip install kaggle"
echo "  2. 配置API token: ~/.kaggle/kaggle.json"
echo ""
echo "下载命令:"
echo "  kaggle datasets download -d ipythonx/mvtec-ad"
echo "  unzip mvtec-ad.zip"
echo ""

# 方式2: 直接下载链接 (社区镜像)
echo "方式2: 社区镜像 (如果可用)"
echo "wget https://www.mydrive.ch/swisscomix/6e8b12c570bb4ac89d5e5a4f7c6c4bb5"
echo ""

# 方式3: 官方申请
echo "方式3: 官方申请 (最稳定)"
echo "访问: https://www.mvtec.com/company/research/datasets/mvtec-ad"
echo "填写申请表，获得下载链接"
echo ""

# 方式4: HuggingFace (推荐，最简单)
echo "方式4: HuggingFace (推荐 ★)"
echo "使用 HuggingFace datasets:"
echo ""
cat > download_from_hf.py << 'EOF'
"""
从HuggingFace下载MVTec AD数据集
"""
from datasets import load_dataset
import os
from pathlib import Path
from PIL import Image
import numpy as np

def download_mvtec_hf(save_dir="./mvtec_anomaly_detection"):
    """从HuggingFace下载MVTec AD"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("正在从HuggingFace下载MVTec AD数据集...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    # 加载数据集
    try:
        dataset = load_dataset("shreyashankar/mvtec-ad", split="train")
        print(f"✓ 训练集加载完成: {len(dataset)} 张图片")
        
        # 保存训练集
        for item in dataset:
            category = item['category']
            image_type = item['type']  # 'good' or defect type
            image = item['image']
            
            cat_dir = save_dir / category / 'train' / image_type
            cat_dir.mkdir(exist_ok=True, parents=True)
            
            image.save(cat_dir / f"{item.get('index', 0):03d}.png")
        
        print(f"✓ 训练集保存完成")
        
        # 加载测试集
        test_dataset = load_dataset("shreyashankar/mvtec-ad", split="test")
        print(f"✓ 测试集加载完成: {len(test_dataset)} 张图片")
        
        # 保存测试集
        for item in test_dataset:
            category = item['category']
            image_type = item['type']
            image = item['image']
            
            cat_dir = save_dir / category / 'test' / image_type
            cat_dir.mkdir(exist_ok=True, parents=True)
            
            image.save(cat_dir / f"{item.get('index', 0):03d}.png")
        
        print(f"✓ 测试集保存完成")
        print(f"\n数据集已保存到: {save_dir.absolute()}")
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 安装datasets库: pip install datasets")
        print("2. 检查网络连接")
        print("3. 使用官方下载方式")

if __name__ == "__main__":
    download_mvtec_hf()
EOF

echo "运行命令: python download_from_hf.py"
echo ""

# 检查是否已有数据集
echo "=========================================="
echo "检查现有数据集..."
echo "=========================================="

if [ -d "bottle" ]; then
    echo "✓ 检测到bottle类别，数据集可能已存在"
    echo "类别列表:"
    ls -d */
else
    echo "✗ 数据集未找到"
    echo ""
    echo "请选择上述方式之一下载数据集:"
    echo "推荐方式4 (最简单): 运行 python download_from_hf.py"
fi

echo ""
echo "=========================================="
