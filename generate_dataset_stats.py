#!/usr/bin/env python3
"""
数据集统计脚本 — 生成论文所需的数据集统计表
输出: experiment_data/dataset_stats/
"""

import os
import json
from pathlib import Path
from collections import defaultdict


def count_images(dir_path):
    """统计目录下图片数量"""
    if not os.path.exists(dir_path):
        return 0
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    return len([f for f in os.listdir(dir_path) 
                if os.path.splitext(f)[1] in exts])


# ============================================
# 1. MVTec AD 统计
# ============================================
def mvtec_stats(data_root):
    """
    统计 MVTec AD 数据集
    每个品类: train/good, test/good, test/defect (各类型)
    """
    text_eval = {}
    object_eval = {}
    
    categories = {
        'carpet': '纹理', 'grid': '纹理', 'leather': '纹理', 
        'tile': '纹理', 'wood': '纹理',
        'bottle': '物体', 'cable': '物体', 'capsule': '物体',
        'hazelnut': '物体', 'metal_nut': '物体', 'pill': '物体',
        'screw': '物体', 'toothbrush': '物体', 'transistor': '物体', 'zipper': '物体'
    }
    
    all_stats = {}
    
    for cat in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        
        train_good = count_images(os.path.join(cat_dir, 'train', 'good'))
        test_good = count_images(os.path.join(cat_dir, 'test', 'good'))
        
        # test/defect 下各类型
        defect_dir = os.path.join(cat_dir, 'test')
        defect_types = {}
        total_defect = 0
        for dtype in os.listdir(defect_dir):
            if dtype == 'good':
                continue
            dtype_path = os.path.join(defect_dir, dtype)
            if os.path.isdir(dtype_path):
                cnt = count_images(dtype_path)
                defect_types[dtype] = cnt
                total_defect += cnt
        
        cat_type = categories.get(cat, '物体')
        all_stats[cat] = {
            'type': cat_type,
            'train_good': train_good,
            'test_good': test_good,
            'test_defect_total': total_defect,
            'defect_types': defect_types,
            'test_total': test_good + total_defect
        }
    
    return all_stats


# ============================================
# 2. 晶圆数据集统计
# ============================================
def wafer_stats(data_root):
    """
    统计晶圆分类数据集
    每个品类: train/good, test/good, test/defect
    """
    all_stats = {}
    
    for cat in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        
        train_good = count_images(os.path.join(cat_dir, 'train', 'good'))
        
        # 按视图统计 test 集
        test_good_dir = os.path.join(cat_dir, 'test', 'good')
        test_defect_dir = os.path.join(cat_dir, 'test', 'defect')
        
        # 统计每个视图
        views = {'UP': {'good': 0, 'defect': 0}, 
                 'DOWN': {'good': 0, 'defect': 0}, 
                 'ALL': {'good': 0, 'defect': 0}}
        
        for vname, vkey in [('good', 'good'), ('defect', 'defect')]:
            d = os.path.join(cat_dir, 'test', vname)
            if os.path.exists(d):
                for f in os.listdir(d):
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        continue
                    if '_UP' in f or '_UP.' in f or f.endswith('_UP.jpg'):
                        views['UP'][vkey] += 1
                    elif '_DOWN' in f or '_DOWN.' in f or f.endswith('_DOWN.jpg'):
                        views['DOWN'][vkey] += 1
                    else:
                        views['ALL'][vkey] += 1
        
        # 清理全零视图
        clean_views = {}
        for v, counts in views.items():
            if counts['good'] > 0 or counts['defect'] > 0:
                clean_views[v] = counts
        
        all_stats[cat] = {
            'train_good': train_good,
            'views': clean_views
        }
    
    return all_stats


# ============================================
# 3. 输出 Markdown 表格
# ============================================
def print_mvtec_md(stats):
    """打印 MVTec 统计为 Markdown"""
    print("\n## MVTec AD 数据集统计\n")
    print("| 品类 | 类型 | train/good | test/good | test/defect | test总计 | 缺陷类型数 |")
    print("|------|------|------------|-----------|-------------|----------|-----------|")
    
    for cat, info in stats.items():
        n_types = len(info['defect_types'])
        print(f"| {cat:12s} | {info['type']:4s} | "
              f"{info['train_good']:10d} | {info['test_good']:9d} | "
              f"{info['test_defect_total']:11d} | {info['test_total']:8d} | {n_types:9d} |")
    
    # 汇总
    total_train = sum(s['train_good'] for s in stats.values())
    total_test_good = sum(s['test_good'] for s in stats.values())
    total_defect = sum(s['test_defect_total'] for s in stats.values())
    print(f"| **合计** | | **{total_train}** | **{total_test_good}** | "
          f"**{total_defect}** | **{total_test_good+total_defect}** | |")


def print_mvtec_defect_detail_md(stats):
    """打印 MVTec 各品类缺陷类型详情"""
    print("\n### 各品类缺陷类型详情\n")
    for cat, info in stats.items():
        if not info['defect_types']:
            continue
        types_str = ', '.join(f"{t}({n})" for t, n in sorted(info['defect_types'].items()))
        print(f"- **{cat}**: {types_str}")


def print_wafer_md(stats):
    """打印晶圆统计为 Markdown"""
    print("\n## 晶圆分类数据集统计\n")
    print("| 品类 | train/good | 视图 | test/good | test/defect |")
    print("|------|------------|------|-----------|-------------|")
    
    for cat, info in stats.items():
        for vname, vcounts in info['views'].items():
            print(f"| {cat:18s} | {info['train_good']:10d} | "
                  f"{vname:4s} | {vcounts['good']:9d} | {vcounts['defect']:11d} |")
    
    # 按视图汇总
    for vname in ['UP', 'DOWN', 'ALL']:
        total_train = sum(s['train_good'] for s in stats.values())
        total_good = sum(s['views'].get(vname, {}).get('good', 0) for s in stats.values())
        total_defect = sum(s['views'].get(vname, {}).get('defect', 0) for s in stats.values())
        if total_good + total_defect > 0:
            print(f"| **{vname}视图合计** | **{total_train}** | | "
                  f"**{total_good}** | **{total_defect}** |")


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    import sys
    
    out_dir = Path('experiment_data/dataset_stats')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # MVTec
    mvtec_root = 'mvtec_anomaly_detection'
    if os.path.exists(mvtec_root):
        mvtec = mvtec_stats(mvtec_root)
        
        # 保存 JSON
        with open(out_dir / 'mvtec_stats.json', 'w', encoding='utf-8') as f:
            json.dump(mvtec, f, ensure_ascii=False, indent=2)
        
        # 保存 Markdown
        with open(out_dir / 'dataset_stats.md', 'w', encoding='utf-8') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            print_mvtec_md(mvtec)
            print_mvtec_defect_detail_md(mvtec)
            sys.stdout = old_stdout
        
        print(f"✅ MVTec AD: 15品类 → {out_dir / 'mvtec_stats.json'} + .md")
    else:
        print(f"⚠️ MVTec 目录不存在: {mvtec_root}")
    
    # 晶圆
    wafer_root = 'data/晶圆分类数据集'
    if os.path.exists(wafer_root):
        wafer = wafer_stats(wafer_root)
        
        with open(out_dir / 'wafer_stats.json', 'w', encoding='utf-8') as f:
            json.dump(wafer, f, ensure_ascii=False, indent=2)
        
        with open(out_dir / 'dataset_stats.md', 'a', encoding='utf-8') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            print_wafer_md(wafer)
            sys.stdout = old_stdout
        
        print(f"✅ 晶圆: 8品类 → {out_dir / 'wafer_stats.json'} + .md")
    else:
        print(f"⚠️ 晶圆目录不存在: {wafer_root}")
    
    # 打印到屏幕
    print("\n" + "="*60)
    if 'mvtec' in dir():
        print_mvtec_md(mvtec)
    if 'wafer' in dir():
        print_wafer_md(wafer)
