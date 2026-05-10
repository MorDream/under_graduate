"""
MVTec AD 数据集下载脚本 (HuggingFace版本)
最简单的方式获取数据集
"""
from datasets import load_dataset
import os
from pathlib import Path
from PIL import Image
import shutil


def download_mvtec_hf(save_dir="./mvtec_anomaly_detection"):
    """
    从HuggingFace下载MVTec AD数据集
    
    安装依赖:
        pip install datasets
    
    使用方法:
        python download_mvtec.py
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("MVTec AD 数据集下载")
    print("=" * 60)
    print(f"保存路径: {save_dir.absolute()}")
    print("")
    
    try:
        print("正在从HuggingFace下载...")
        print("(此过程可能需要5-10分钟，取决于网络速度)")
        print("")
        
        # 加载训练集
        print("[1/3] 加载训练集...")
        train_dataset = load_dataset("shreyashankar/mvtec-ad", split="train", streaming=False)
        
        # 加载测试集
        print("[2/3] 加载测试集...")
        test_dataset = load_dataset("shreyashankar/mvtec-ad", split="test", streaming=False)
        
        # 处理并保存
        print("[3/3] 保存到本地...")
        
        # 保存训练集 - 只保存good类别
        train_count = 0
        for item in train_dataset:
            category = item['category']
            image_type = item['type']
            image = item['image']
            
            # 只保存正常样本(good)到train目录
            if image_type == 'good':
                cat_dir = save_dir / category / 'train' / 'good'
                cat_dir.mkdir(exist_ok=True, parents=True)
                image.save(cat_dir / f"{train_count:03d}.png")
                train_count += 1
        
        print(f"  ✓ 训练集: {train_count} 张正常样本")
        
        # 保存测试集 - 包含good和所有defect类型
        test_count = {'good': 0}
        for item in test_dataset:
            category = item['category']
            image_type = item['type']
            image = item['image']
            
            cat_dir = save_dir / category / 'test' / image_type
            cat_dir.mkdir(exist_ok=True, parents=True)
            
            idx = test_count.get(image_type, 0)
            image.save(cat_dir / f"{idx:03d}.png")
            test_count[image_type] = idx + 1
        
        print(f"  ✓ 测试集统计:")
        for k, v in test_count.items():
            print(f"    - {k}: {v} 张")
        
        # 列出所有类别
        categories = [d.name for d in save_dir.iterdir() if d.is_dir()]
        print(f"\n✓ 数据集下载完成!")
        print(f"✓ 共 {len(categories)} 个类别:")
        for cat in sorted(categories):
            print(f"  - {cat}")
        
        print(f"\n数据集路径: {save_dir.absolute()}")
        print("")
        print("使用方法:")
        print(f"  python train_improved_v3.py --dataset mvtec --mvtec_category bottle")
        
        return True
        
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("\n请安装:")
        print("  pip install datasets")
        return False
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n其他下载方式:")
        print("1. 官方: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print("2. Kaggle: https://www.kaggle.com/datasets/ipythonx/mvtec-ad")
        return False


def verify_dataset(data_dir="./mvtec_anomaly_detection"):
    """验证数据集完整性"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"✗ 数据集目录不存在: {data_dir}")
        return False
    
    categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    print("\n数据集验证:")
    print(f"  类别数: {len(categories)}")
    
    for cat in sorted(categories)[:3]:  # 只显示前3个
        cat_dir = data_dir / cat
        train_dir = cat_dir / 'train' / 'good'
        test_dir = cat_dir / 'test'
        
        train_count = len(list(train_dir.glob('*.png'))) if train_dir.exists() else 0
        test_count = sum(len(list(d.glob('*.png'))) for d in test_dir.iterdir()) if test_dir.exists() else 0
        
        print(f"  {cat:15s}: train={train_count:3d}, test={test_count:3d}")
    
    if len(categories) > 3:
        print(f"  ... 还有 {len(categories)-3} 个类别")
    
    return len(categories) > 0


if __name__ == "__main__":
    import sys
    
    save_dir = sys.argv[1] if len(sys.argv) > 1 else "./mvtec_anomaly_detection"
    
    # 先验证是否已有数据集
    if verify_dataset(save_dir):
        print("\n✓ 数据集已存在，无需重新下载")
        response = input("\n是否重新下载? (y/N): ")
        if response.lower() != 'y':
            print("使用现有数据集")
            exit(0)
    
    # 下载数据集
    success = download_mvtec_hf(save_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("下载成功! 开始训练示例:")
        print("=" * 60)
        print(f"\n# 训练单个类别:")
        print(f"python train_improved_v3.py \\")
        print(f"    --dataset mvtec \\")
        print(f"    --mvtec_category bottle \\")
        print(f"    --mvtec_dir {save_dir} \\")
        print(f"    --epochs 100 \\")
        print(f"    --batch_size 32")
        print(f"\n# 训练并评估所有类别:")
        print(f"python train_improved_v3.py \\")
        print(f"    --mode train_eval_all \\")
        print(f"    --dataset mvtec \\")
        print(f"    --mvtec_dir {save_dir} \\")
        print(f"    --epochs 100")
