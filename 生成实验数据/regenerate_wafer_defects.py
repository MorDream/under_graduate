#!/usr/bin/env python3
"""
重新生成晶圆合成缺陷（UP浓/DOWN适中）
先删除所有 syn_ 文件，再按新参数重新生成
"""
import os
import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generate_synthetic_defects import (
    imread_unicode, imwrite_unicode, 
    add_scratch, add_stains, add_spots, add_missing_region, add_texture_noise
)
import numpy as np

# DOWN 用适中参数（函数默认值）
DOWN_PARAMS = {
    'scratch': {'max_scratches': 2, 'min_width': 1, 'max_width': 2, 'min_opacity': 0.18, 'max_opacity': 0.38},
    'stains': {'max_stains': 2, 'min_size': 8, 'max_size': 30, 'min_opacity': 0.12, 'max_opacity': 0.28},
    'spots': {'max_spots': 6, 'spot_size': 2, 'opacity_range': (0.15, 0.30)},
    'missing': {'max_regions': 1, 'min_ratio': 0.03, 'max_ratio': 0.10},
}

# UP 用浓参数
UP_PARAMS = {
    'scratch': {'max_scratches': 3, 'min_width': 2, 'max_width': 4, 'min_opacity': 0.45, 'max_opacity': 0.75},
    'stains': {'max_stains': 3, 'min_size': 12, 'max_size': 50, 'min_opacity': 0.35, 'max_opacity': 0.55},
    'spots': {'max_spots': 12, 'spot_size': 3, 'opacity_range': (0.40, 0.65)},
    'missing': {'max_regions': 2, 'min_ratio': 0.05, 'max_ratio': 0.18},
}


def detect_view(filename):
    """从文件名检测 UP/DOWN"""
    stem = Path(filename).stem
    if '_UP' in stem or stem.endswith('UP'):
        return 'UP'
    return 'DOWN'


def apply_all_defects(img, is_up=False):
    """应用所有缺陷类型"""
    params = UP_PARAMS if is_up else DOWN_PARAMS
    result = img.copy()
    result = add_scratch(result, **params['scratch'])
    result = add_stains(result, **params['stains'])
    result = add_spots(result, **params['spots'])
    result = add_missing_region(result, **params['missing'])
    result = add_texture_noise(result)
    return result


def delete_existing_synthetic(data_root, category=None):
    """删除现有的 syn_ 文件"""
    data_root = Path(data_root)
    categories = [category] if category else [d.name for d in data_root.iterdir() if d.is_dir()]
    
    deleted = 0
    for cat in categories:
        defect_dir = data_root / cat / 'test' / 'defect'
        if not defect_dir.exists():
            continue
        for f in defect_dir.glob('syn_*.jpg'):
            f.unlink()
            deleted += 1
    return deleted


def regenerate_category(data_root, category, variants_per_image=2):
    """重新生成单个品类的合成缺陷"""
    cat_dir = Path(data_root) / category
    good_dir = cat_dir / 'test' / 'good'
    defect_dir = cat_dir / 'test' / 'defect'
    
    if not good_dir.exists():
        return 0
    
    good_files = sorted([f for f in good_dir.iterdir() 
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    
    if not good_files:
        return 0
    
    # 分离 UP/DOWN
    up_files = [f for f in good_files if detect_view(f.name) == 'UP']
    down_files = [f for f in good_files if detect_view(f.name) == 'DOWN']
    
    print(f"  {category}: good总数={len(good_files)}, UP={len(up_files)}, DOWN={len(down_files)}")
    
    generated = 0
    img_idx = 0
    
    # 每张 good 图生成 variants_per_image 个变体
    for src_file in good_files:
        is_up = detect_view(src_file.name) == 'UP'
        view_tag = 'UP_HEAVY' if is_up else 'DOWN_MEDIUM'
        
        img = imread_unicode(str(src_file))
        if img is None:
            continue
        
        for v in range(variants_per_image):
            np.random.seed(img_idx * 100 + v * 17)
            random.seed(img_idx * 100 + v * 17)
            
            result = apply_all_defects(img, is_up=is_up)
            syn_name = f"syn_{view_tag}_{img_idx:04d}_v{v}_{src_file.stem}.jpg"
            syn_path = defect_dir / syn_name
            imwrite_unicode(str(syn_path), result)
            generated += 1
        
        img_idx += 1
    
    print(f"    → 生成了 {generated} 张 (UP_HEAVY + DOWN_MEDIUM)")
    return generated


def main():
    parser = argparse.ArgumentParser(description='重新生成晶圆合成缺陷（UP浓/DOWN适中）')
    parser.add_argument('--data_root', type=str, default='data/晶圆分类数据集')
    parser.add_argument('--category', type=str, default=None, help='指定品类')
    parser.add_argument('--variants', type=int, default=2, help='每张good图生成变体数')
    parser.add_argument('--skip-delete', action='store_true', help='跳过删除步骤')
    args = parser.parse_args()
    
    print("=" * 60)
    print("晶圆合成缺陷重新生成")
    print(f"UP视图: 浓参数 (划痕0.45-0.75, 污渍0.35-0.55)")
    print(f"DOWN视图: 适中参数 (划痕0.18-0.38, 污渍0.12-0.28)")
    print("=" * 60)
    
    # 步骤1: 删除现有合成缺陷
    if not args.skip_delete:
        print("\n[1/2] 删除现有 syn_ 文件...")
        deleted = delete_existing_synthetic(args.data_root, args.category)
        print(f"  删除了 {deleted} 个旧文件")
    
    # 步骤2: 重新生成
    print("\n[2/2] 重新生成合成缺陷...")
    data_root = Path(args.data_root)
    categories = [args.category] if args.category else sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    
    total = 0
    for cat in categories:
        n = regenerate_category(args.data_root, cat, args.variants)
        total += n
    
    print(f"\n{'=' * 60}")
    print(f"✅ 完成！共生成 {total} 张合成缺陷")
    print(f"{'=' * 60}")
    
    # 统计
    print("\n最终分布:")
    for cat in categories:
        cat_dir = data_root / cat
        good = len(list((cat_dir / 'test' / 'good').glob('*')))
        real_defect = len([f for f in (cat_dir / 'test' / 'defect').glob('*') if not f.name.startswith('syn_')])
        syn_defect = len([f for f in (cat_dir / 'test' / 'defect').glob('syn_*.jpg')])
        up_syn = len([f for f in (cat_dir / 'test' / 'defect').glob('*') if 'UP_HEAVY' in f.name])
        down_syn = len([f for f in (cat_dir / 'test' / 'defect').glob('*') if 'DOWN_MEDIUM' in f.name])
        print(f"  {cat:20s} good={good:3d} real_defect={real_defect:3d} syn={syn_defect:3d} (UP={up_syn},DOWN={down_syn})")


if __name__ == '__main__':
    main()
