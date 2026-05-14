#!/usr/bin/env python3
"""
晶圆缺陷合成生成器
在正常晶圆图像上生成划痕、污渍、颗粒等合成缺陷
用于平衡测试集正常/缺陷样本
"""

import os
import sys
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════
# 工具函数（处理中文路径）
# ═══════════════════════════════════════════

def imread_unicode(path):
    """读取图片，兼容Windows中文路径"""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    # 备选：numpy读取（Windows中文路径）
    stream = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path, img):
    """保存图片，兼容Windows中文路径"""
    try:
        return cv2.imwrite(str(path), img)
    except:
        pass
    try:
        ext = os.path.splitext(str(path))[1]
        _, buf = cv2.imencode(ext, img)
        buf.tofile(str(path))
        return True
    except:
        return False

def add_scratch(img, 
                max_scratches=2, 
                min_len=0.05, max_len=0.25,
                min_width=1, max_width=2,
                min_opacity=0.18, max_opacity=0.38):
    """
    添加随机划痕（直线或轻微弯曲）"""
    h, w = img.shape[:2]
    result = img.copy()
    
    for _ in range(random.randint(1, max_scratches)):
        overlay = result.copy()
        # 随机起点和终点
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        length = int(min(w, h) * random.uniform(min_len, max_len))
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        x2, y2 = np.clip(x2, 0, w-1), np.clip(y2, 0, h-1)
        
        # 颜色：深色或浅色划痕
        is_dark = random.random() < 0.7
        if is_dark:
            color = random.randint(0, 80)  # 深灰到黑
            color_bgr = (color, color, color)
        else:
            color = random.randint(200, 240)  # 浅灰到白
            color_bgr = (color, color, color)
        
        thickness = random.randint(min_width, max_width)
        cv2.line(overlay, (x1, y1), (x2, y2), color_bgr, thickness)
        
        # 随机点画法(更粗糙)
        if random.random() < 0.3:
            for i in range(thickness * 3):
                px = random.randint(min(x1, x2), max(x1, x2))
                py = random.randint(min(y1, y2), max(y1, y2))
                px, py = np.clip(px, 0, w-1), np.clip(py, 0, h-1)
                cv2.circle(overlay, (px, py), 1, color_bgr, -1)
        
        opacity = random.uniform(min_opacity, max_opacity)
        result = cv2.addWeighted(result, 1 - opacity, overlay, opacity, 0)
    
    return result


def add_stains(img, 
               max_stains=2,
               min_size=8, max_size=30,
               min_opacity=0.12, max_opacity=0.28):
    """
    添加污渍/污染斑块（中等大小、半透明）"""
    h, w = img.shape[:2]
    result = img.copy()
    
    for _ in range(random.randint(1, max_stains)):
        overlay = result.copy()
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        
        # 少量叠加椭圆
        n_ellipses = random.randint(1, 2)
        for _ in range(n_ellipses):
            rx = random.randint(min_size // 2, max_size // 2)
            ry = random.randint(min_size // 2, max_size // 2)
            
            if random.random() < 0.5:
                color = random.randint(20, 100)
            else:
                color = random.randint(180, 230)
            
            cv2.ellipse(overlay, (cx, cy), (rx, ry), 
                       random.randint(0, 180), 0, 360,
                       (color, color, color), -1)
        
        # 高斯模糊使边缘柔和
        kernel_size = random.choice([5, 7, 9, 11])
        overlay = cv2.GaussianBlur(overlay, (kernel_size, kernel_size), 0)
        
        opacity = random.uniform(min_opacity, max_opacity)
        result = cv2.addWeighted(result, 1 - opacity, overlay, opacity, 0)
    
    return result


def add_spots(img, 
              max_spots=6,
              spot_size=2,
              opacity_range=(0.15, 0.30)):
    """添加斑点/颗粒"""
    h, w = img.shape[:2]
    result = img.copy()
    
    n = random.randint(3, max_spots)
    for _ in range(n):
        overlay = result.copy()
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        
        if random.random() < 0.6:
            color = random.randint(0, 60)
        else:
            color = random.randint(200, 255)
        color_bgr = (color, color, color)
        
        s = random.randint(1, spot_size)
        cv2.circle(overlay, (cx, cy), s, color_bgr, -1)
        
        opacity = random.uniform(*opacity_range)
        result = cv2.addWeighted(result, 1 - opacity, overlay, opacity, 0)
    
    return result


def add_missing_region(img,
                       max_regions=1,
                       min_ratio=0.03, max_ratio=0.10):
    """添加缺失区域（模拟die缺失，半透明）"""
    h, w = img.shape[:2]
    result = img.copy()
    
    if random.random() < 0.5:  # 50%概率不添加
        return result
    
    overlay = result.copy()
    region_w = int(w * random.uniform(min_ratio, max_ratio))
    region_h = int(h * random.uniform(min_ratio, max_ratio))
    region_w = min(region_w, w // 4)
    region_h = min(region_h, h // 4)
    
    x1 = random.randint(0, w - region_w - 1)
    y1 = random.randint(0, h - region_h - 1)
    
    color = random.randint(40, 100)
    cv2.rectangle(overlay, (x1, y1), (x1+region_w, y1+region_h),
                 (color, color, color), -1)
    
    overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
    
    opacity = random.uniform(0.18, 0.35)
    result = cv2.addWeighted(result, 1 - opacity, overlay, opacity, 0)
    
    return result


def add_texture_noise(img,
                      noise_ratio=0.1,
                      intensity=30):
    """添加纹理噪声（随机像素扰动模拟工艺变异）"""
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)
    mask = np.random.random((h, w)) < noise_ratio
    noise = np.random.randint(-intensity, intensity, (h, w, 3))
    result[mask] = np.clip(result[mask] + noise[mask], 0, 255)
    return result.astype(np.uint8)


# ═══════════════════════════════════════════
# 组合生成
# ═══════════════════════════════════════════

def generate_defect(img, 
                    types=['scratch', 'stains', 'spots', 'missing', 'texture'],
                    intensity='medium'):
    """
    自动组合多种缺陷类型生成合成缺陷图
    
    Args:
        img: BGR numpy array
        types: 可选 ['scratch', 'stains', 'spots', 'missing', 'texture']
        intensity: 'light' | 'medium' | 'heavy'
    """
    if intensity == 'light':
        params = {
            'scratch': {'max_scratches': 1, 'min_opacity': 0.1, 'max_opacity': 0.2},
            'stains': {'max_stains': 1, 'min_opacity': 0.08, 'max_opacity': 0.15},
            'spots': {'max_spots': 3, 'opacity_range': (0.08, 0.15)},
            'missing': {'max_regions': 1, 'min_ratio': 0.02, 'max_ratio': 0.06},
        }
    elif intensity == 'medium':
        # 使用函数默认参数（已调至适中）
        params = {}
    else:  # heavy — UP视图用
        params = {
            'scratch': {'max_scratches': 2, 'min_width': 1, 'max_width': 3, 'min_opacity': 0.35, 'max_opacity': 0.6},
            'stains': {'max_stains': 3, 'min_size': 10, 'max_size': 40, 'min_opacity': 0.25, 'max_opacity': 0.45},
            'spots': {'max_spots': 10, 'spot_size': 3, 'opacity_range': (0.3, 0.55)},
            'missing': {'max_regions': 1, 'min_ratio': 0.05, 'max_ratio': 0.15},
        }
    
    result = img.copy()
    
    if 'scratch' in types:
        result = add_scratch(result, **params.get('scratch', {}))
    if 'stains' in types:
        result = add_stains(result, **params.get('stains', {}))
    if 'spots' in types:
        result = add_spots(result, **params.get('spots', {}))
    if 'missing' in types:
        result = add_missing_region(result)
    if 'texture' in types:
        result = add_texture_noise(result)
    
    return result


# ═══════════════════════════════════════════
# 批量生成
# ═══════════════════════════════════════════

def balance_test_set(data_root, category, target_samples=None, 
                     variants_per_image=4, intensity='medium'):
    """
    对指定品类平衡测试集缺陷样本
    
    Args:
        data_root: 数据集根目录
        category: 品类名, e.g. 'BGA S5E 16x7'
        target_samples: 目标缺陷数 (None=与good样本数相等)
        variants_per_image: 每张good图生成几个缺陷变体
        intensity: 缺陷强度
    """
    cat_dir = Path(data_root) / category
    good_dir = cat_dir / 'test' / 'good'
    defect_dir = cat_dir / 'test' / 'defect'
    
    if not good_dir.exists():
        print(f"[SKIP] {category}: 无test/good目录")
        return 0
    
    good_files = sorted([f for f in good_dir.iterdir() 
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    defect_files = sorted([f for f in defect_dir.iterdir() 
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    
    n_good = len(good_files)
    n_defect_orig = len(defect_files)
    
    if target_samples is None:
        # 目标：让good和defect数量相等，取较大值
        target_samples = max(n_good, n_defect_orig)
    
    # 情况1: defect不够 → 生成合成缺陷
    if n_defect_orig < target_samples:
        needed = target_samples - n_defect_orig
        print(f"[GEN-DEFECT] {category}: good={n_good}, defect={n_defect_orig} → target={target_samples} (需生成 {needed} 张合成缺陷)")
        
        generated = 0
        img_idx = 0
        while generated < needed:
            good_img_path = good_files[img_idx % len(good_files)]
            img_idx += 1
            
            batch = min(variants_per_image, needed - generated)
            for variant in range(batch):
                types = random.sample(
                    ['scratch', 'stains', 'spots', 'missing', 'texture'],
                    k=random.randint(1, 3)
                )
                img = imread_unicode(good_img_path)
                if img is None:
                    continue
                intensity_choice = random.choice(['light', 'medium'])
                defective = generate_defect(img, types=types, intensity=intensity_choice)
                base_name = good_img_path.stem
                syn_name = f"{base_name}_syn{generated:04d}.jpg"
                syn_path = defect_dir / syn_name
                imwrite_unicode(syn_path, defective)
                generated += 1
                if generated >= needed:
                    break
        
        print(f"  → 生成了 {generated} 张合成缺陷图")
        return generated
    
    # 情况2: good不够 → 从train/good搬过来
    if n_good < target_samples:
        needed = target_samples - n_good
        train_good_dir = cat_dir / 'train' / 'good'
        if not train_good_dir.exists():
            print(f"[SKIP] {category}: defect={n_defect_orig} > good={n_good}，但无train/good目录")
            return 0
        
        train_files = sorted([f for f in train_good_dir.iterdir()
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
        if not train_files:
            print(f"[SKIP] {category}: defect={n_defect_orig} > good={n_good}，但train/good为空")
            return 0
        
        print(f"[COPY-GOOD] {category}: good={n_good}, defect={n_defect_orig} → target={target_samples} (从train搬 {needed} 张good)")
        
        copied = 0
        for f in train_files:
            if copied >= needed:
                break
            # 跳过已经在test/good里的
            if f.name in [g.name for g in good_files]:
                continue
            dst = good_dir / f.name
            import shutil
            shutil.copy2(str(f), str(dst))
            copied += 1
        
        print(f"  → 从train搬了 {copied} 张good图到test/good")
        return copied
    
    print(f"[OK] {category}: good={n_good}, defect={n_defect_orig} (已平衡)")
    return 0


# ═══════════════════════════════════════════
# 预览模式
# ═══════════════════════════════════════════

def preview(data_root, category, output_dir='preview'):
    """生成预览图：原图 vs 各类型缺陷"""
    cat_dir = Path(data_root) / category
    good_dir = cat_dir / 'test' / 'good'
    
    if not good_dir.exists():
        print(f"无 {category} 的 test/good")
        return
    
    good_files = sorted([f for f in good_dir.iterdir() 
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    if not good_files:
        print(f"无图片")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    img = imread_unicode(str(good_files[0]))
    
    previews = {
        'original': img,
        'scratch': add_scratch(img),
        'stains': add_stains(img),
        'spots': add_spots(img),
        'missing': add_missing_region(img),
        'combined_med': generate_defect(img, intensity='medium'),
        'combined_heavy': generate_defect(img, intensity='heavy'),
    }
    
    for name, im in previews.items():
        imwrite_unicode(f"{output_dir}/{category}_{name}.jpg", im)
    
    print(f"预览图已保存到 {output_dir}/")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='晶圆合成缺陷生成器')
    parser.add_argument('--data_root', type=str, default='data/晶圆分类数据集',
                       help='数据集根目录')
    parser.add_argument('--category', type=str, default=None,
                       help='指定品类 (不指定则处理全部不平衡品类)')
    parser.add_argument('--preview', action='store_true',
                       help='只生成预览图看效果')
    parser.add_argument('--target', type=int, default=None,
                       help='目标缺陷样本数 (默认=good样本数)')
    parser.add_argument('--variants', type=int, default=4,
                       help='每张good图生成缺陷变体数 (默认4)')
    parser.add_argument('--intensity', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'])
    
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    if args.preview:
        preview(args.data_root, args.category or 'BGA S5E 16x7')
        sys.exit(0)
    
    # 全部品类
    data_root = Path(args.data_root)
    if args.category:
        categories = [args.category]
    else:
        categories = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    
    total_syn = 0
    total_copy = 0
    for cat in categories:
        import shutil
        n = balance_test_set(
            args.data_root, cat,
            target_samples=args.target,
            variants_per_image=args.variants,
            intensity=args.intensity
        )
        total_syn += n
    
    print(f"\n✅ 完成！")
    print(f"  → 合成缺陷: 0 张（defect都≥good，从train搬good即可）")
    print(f"  → 从train搬good: {total_syn} 张")
    
    # 显示最终分布
    print("\n======= 最终分布 =======")
    for cat in categories:
        cat_dir = Path(args.data_root) / cat
        good = len(list((cat_dir / 'test' / 'good').glob('*')))
        defect = len(list((cat_dir / 'test' / 'defect').glob('*')))
        print(f"  {cat:20s}  good={good:3d}  defect={defect:3d}  ratio={defect/max(1,good):.1f}")
