#!/usr/bin/env python3
"""
生成论文需要的可视化示例图:
- CutPaste 增强前后对比
- 合成缺陷类型示例
- 正常/异常热力图样本 (placeholder，等训练完后跑)
输出: experiment_data/visualization/
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
from recontrast.dataset import cut_paste, get_data_transforms
import torch
import torchvision.transforms as T


def make_comparison_grid(images, titles, output_path, cols=3):
    """生成对比网格图"""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # 统一大小
    thumb_size = 256
    thumbnails = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # 反归一化
            img = img.clone()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            img = torch.clamp(img, 0, 1)
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img)
        elif isinstance(img, np.ndarray):
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        thumbnails.append(img)
    
    # 创建画布
    canvas = Image.new('RGB', (thumb_size * cols + 20 * (cols + 1), 
                                thumb_size * rows + 40 * rows + 20), 'white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        font = ImageFont.load_default()
    
    for i, (img, title) in enumerate(zip(thumbnails, titles)):
        row, col = i // cols, i % cols
        x = 10 + col * (thumb_size + 20)
        y = 10 + row * (thumb_size + 40)
        canvas.paste(img, (x, y + 25))
        draw.text((x + thumb_size // 2 - len(title) * 4, y + 5), title, fill='black', font=font)
    
    canvas.save(output_path, quality=95)
    return output_path


def generate_cutpaste_examples():
    """生成 CutPaste 增强前后对比 (MVTec carpet)"""
    out_dir = Path('experiment_data/visualization')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 找 carpet 训练图
    carpet_dir = Path('mvtec_anomaly_detection/carpet/train/good')
    imgs = sorted(carpet_dir.glob('*.png'))[:4]
    
    if not imgs:
        print("⚠️ 无 carpet 训练图")
        return
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    images = []
    titles = []
    
    for i, img_path in enumerate(imgs[:2]):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        # 原始
        images.append(img_tensor.clone())
        titles.append(f'原图 {i+1}')
        
        # CutPaste
        cp = cut_paste(img_tensor.clone())
        images.append(cp)
        titles.append(f'CutPaste {i+1}')
        
        # 再做一个不同随机种子的
        np.random.seed(i * 42)
        cp2 = cut_paste(img_tensor.clone())
        images.append(cp2)
        titles.append(f'CutPaste {i+1}b')
    
    output = out_dir / 'cutpaste_examples.jpg'
    make_comparison_grid(images, titles, str(output), cols=3)
    print(f"✅ CutPaste 示例: {output}")


def generate_synthetic_defect_examples():
    """生成合成缺陷类型示例"""
    import cv2
    sys.path.insert(0, '.')
    from generate_synthetic_defects import (
        imread_unicode, add_scratch, add_stains, add_spots, 
        add_missing_region, generate_defect
    )
    
    out_dir = Path('experiment_data/visualization')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 找晶圆图
    wafer_dir = Path('data/晶圆分类数据集/BGA S5E 16x7/test/good')
    imgs = sorted(wafer_dir.glob('*.jpg'))[:1]
    
    if not imgs:
        # 尝试其他品类
        for cat_dir in Path('data/晶圆分类数据集').iterdir():
            if cat_dir.is_dir():
                good_dir = cat_dir / 'test' / 'good'
                imgs = sorted(good_dir.glob('*.jpg'))[:1]
                if imgs:
                    break
    
    if not imgs:
        print("⚠️ 无晶圆图片")
        return
    
    img = imread_unicode(str(imgs[0]))
    if img is None:
        print("⚠️ 无法读取晶圆图")
        return
    
    # 生成各类型
    variants = {
        '原始图': img,
        '划痕': add_scratch(img.copy()),
        '污渍': add_stains(img.copy()),
        '斑点': add_spots(img.copy()),
        '缺失': add_missing_region(img.copy()),
        '中度组合': generate_defect(img.copy(), intensity='medium'),
        '重度组合': generate_defect(img.copy(), intensity='heavy'),
    }
    
    # 统一大小并保存
    h, w = img.shape[:2]
    total_w = w * 4
    total_h = h * 2 + 30
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    for i, (name, variant) in enumerate(variants.items()):
        row, col = i // 4, i % 4
        x = col * w
        y = row * (h + 30)
        canvas[y:y+h, x:x+w] = variant
        cv2.putText(canvas, name, (x + 10, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    output = out_dir / 'synthetic_defect_types.jpg'
    cv2.imwrite(str(output), canvas)
    print(f"✅ 合成缺陷类型: {output}")


if __name__ == '__main__':
    generate_cutpaste_examples()
    generate_synthetic_defect_examples()
    print("\n✅ 可视化示例生成完成！保存至 experiment_data/visualization/")
