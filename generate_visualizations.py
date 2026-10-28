#!/usr/bin/env python3
"""
生成论文需要的可视化示例图（修复中文乱码）
- CutPaste 增强前后对比
- 合成缺陷类型示例
输出: experiment_data/visualization/
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
from recontrast.dataset import cut_paste
import torch
import torchvision.transforms as T


# ============================================
# 中文字体检测
# ============================================
def get_chinese_font(size=18):
    """自动检测可用的中文字体"""
    candidates = [
        # WSL/Linux
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        # Windows (WSL挂载)
        '/mnt/c/Windows/Fonts/msyh.ttc',
        '/mnt/c/Windows/Fonts/simhei.ttf',
        '/mnt/c/Windows/Fonts/simsun.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()


def put_chinese_text(img_pil, text, position, font=None, fill='black'):
    """在PIL图像上绘制中文文字"""
    draw = ImageDraw.Draw(img_pil)
    if font is None:
        font = get_chinese_font()
    draw.text(position, text, fill=fill, font=font)


# ============================================
# CutPaste 对比图
# ============================================
def generate_cutpaste_examples():
    """生成 CutPaste 增强前后对比 (MVTec carpet)"""
    out_dir = Path('experiment_data/visualization')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    carpet_dir = Path('mvtec_anomaly_detection/carpet/train/good')
    imgs = sorted(carpet_dir.glob('*.png'))
    if not imgs:
        print("⚠️ 无 carpet 训练图")
        return
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    
    # 选2张图，每张原图 + 2个不同的CutPaste = 6张
    images, titles = [], []
    font = get_chinese_font(16)
    
    for i in range(2):
        img_tensor = transform(Image.open(imgs[i]).convert('RGB'))
        images.append(img_tensor)
        titles.append(f'原图 {i+1}')
        
        np.random.seed(i * 37)
        cp1 = cut_paste(img_tensor.clone())
        images.append(cp1)
        titles.append(f'CutPaste {i+1}a')
        
        np.random.seed(i * 37 + 7)
        cp2 = cut_paste(img_tensor.clone())
        images.append(cp2)
        titles.append(f'CutPaste {i+1}b')
    
    thumb_size = 256
    cols, rows = 3, 2
    canvas = Image.new('RGB', (thumb_size * cols + 30, thumb_size * rows + 50 + 30), 'white')
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row, col = idx // cols, idx % cols
        x = 15 + col * (thumb_size + 10)
        y = 15 + row * (thumb_size + 50)
        
        # 转PIL
        img_np = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        img_np = torch.clamp(img_np, 0, 1)
        img_np = (img_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        canvas.paste(pil_img, (x, y + 30))
        put_chinese_text(canvas, title, (x + thumb_size // 2 - 40, y), font=font)
    
    output = out_dir / 'cutpaste_examples.jpg'
    canvas.save(output, quality=95)
    print(f"✅ CutPaste 示例: {output}")


# ============================================
# 合成缺陷类型
# ============================================
def generate_synthetic_defect_examples():
    """生成合成缺陷类型示例（使用PIL渲染中文）"""
    from generate_synthetic_defects import imread_unicode, add_scratch, add_stains, add_spots, add_missing_region, generate_defect as gen_def
    
    out_dir = Path('experiment_data/visualization')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 找晶圆图
    for cat_dir in sorted(Path('data/晶圆分类数据集').iterdir()):
        if cat_dir.is_dir():
            good_dir = cat_dir / 'test' / 'good'
            imgs = sorted(good_dir.glob('*.jpg'))
            if imgs:
                break
    
    if not imgs:
        print("⚠️ 无晶圆图片")
        return
    
    img = imread_unicode(str(imgs[0]))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    variants = [
        ('原图', img),
        ('划痕 (Scratch)', add_scratch(img.copy())),
        ('污渍 (Stain)', add_stains(img.copy())),
        ('斑点 (Spot)', add_spots(img.copy())),
        ('缺失 (Missing)', add_missing_region(img.copy())),
        ('中度组合', gen_def(img.copy(), intensity='medium')),
        ('重度组合', gen_def(img.copy(), intensity='heavy')),
    ]
    
    # 3列x3行网格
    cols = 3
    rows = (len(variants) + cols - 1) // cols
    cell_w, cell_h = 310, 180
    
    canvas = Image.new('RGB', (cell_w * cols + 20, cell_h * rows + 20), 'white')
    font = get_chinese_font(14)
    
    for idx, (name, img_bgr) in enumerate(variants):
        row, col = idx // cols, idx % cols
        x = 10 + col * cell_w
        y = 10 + row * cell_h
        
        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((280, 140))
        
        canvas.paste(pil_img, (x + 10, y + 25))
        put_chinese_text(canvas, name, (x + 10, y + 5), font=font)
    
    output = out_dir / 'synthetic_defect_types.jpg'
    canvas.save(output, quality=95)
    print(f"✅ 合成缺陷类型: {output}")


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    import cv2
    generate_cutpaste_examples()
    generate_synthetic_defect_examples()
    print("✅ 可视化示例生成完成！")
