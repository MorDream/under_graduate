#!/usr/bin/env python3
"""
ViT+ReContrast 消融实验运行脚本
自动运行所有消融实验并汇总结果

消融实验列表:
  Exp0: CutPaste 增强 (ON vs OFF)
  Exp1: 交叉重建 (Cross vs Self)
  Exp2: 难例挖掘 (Hard Mining ON vs OFF)  
  Exp3: 预训练权重 (DINOv2 vs Random)
  Exp4: 输入分辨率 (518 vs 224)
  Exp5: 中间层数 (n=2 vs n=4 vs n=6)
  Baseline: 全开 (所有默认设置)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path


def run_exp(name, extra_args, categories, dataset='mvtec', base_cmd='python recontrast_vit_wafer.py'):
    """运行一次消融实验"""
    cat_str = ','.join(categories)
    cmd = f"{base_cmd} --dataset {dataset} --categories {cat_str} {extra_args} --eval_interval 200"
    print(f"\n{'='*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name}")
    print(f"  {extra_args}")
    print(f"{'='*60}")
    
    start = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start
    
    if ret != 0:
        print(f"  ❌ 失败 (exit={ret})")
        return {'name': name, 'status': 'FAILED', 'time': elapsed}
    
    print(f"  ✅ 完成 ({elapsed:.0f}s)")
    return {'name': name, 'status': 'OK', 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(description='ViT+ReContrast 消融实验')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'wafer'])
    parser.add_argument('--categories', type=str, default='carpet',
                       help='逗号分隔品类 (默认carpet，快速验证)')
    parser.add_argument('--base_dir', type=str, default='./ablation_results_vit',
                       help='结果保存目录')
    parser.add_argument('--all', action='store_true',
                       help='运行全部消融（否则只跑baseline）')
    args = parser.parse_args()
    
    categories = [c.strip() for c in args.categories.split(',')]
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ====================
    # Baseline: 全开
    # ====================
    print(f"\n{'#'*60}")
    print(f"# 消融实验 — ViT+ReContrast ({args.dataset})")
    print(f"# 品类: {categories}")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    baseline_name = f"Baseline_全开"
    baseline_args = f"--save_dir {base_dir}/baseline"
    r = run_exp(baseline_name, baseline_args, categories, args.dataset)
    r['exp_id'] = 'Baseline'
    results.append(r)
    
    if not args.all:
        print("\n提示: 加 --all 运行全部消融实验")
        return
    
    # ====================
    # Exp0: CutPaste ON vs OFF
    # ====================
    r = run_exp("Exp0_CutPaste_OFF", 
                f"--save_dir {base_dir}/exp0_no_cutpaste --ablation_no_cutpaste",
                categories, args.dataset)
    r['exp_id'] = 'Exp0'
    results.append(r)
    
    # ====================
    # Exp1: Cross vs Self Reconstruction
    # ====================
    r = run_exp("Exp1_Self_Recon",
                f"--save_dir {base_dir}/exp1_self_recon --ablation_self_recon",
                categories, args.dataset)
    r['exp_id'] = 'Exp1'
    results.append(r)
    
    # ====================
    # Exp2: Hard Mining ON vs OFF
    # ====================
    r = run_exp("Exp2_No_Hard_Mining",
                f"--save_dir {base_dir}/exp2_no_hard_mining --ablation_no_hard_mining",
                categories, args.dataset)
    r['exp_id'] = 'Exp2'
    results.append(r)
    
    # ====================
    # Exp3: Pretrained vs Random
    # ====================
    r = run_exp("Exp3_Random_Init",
                f"--save_dir {base_dir}/exp3_random_init --ablation_no_pretrained",
                categories, args.dataset)
    r['exp_id'] = 'Exp3'
    results.append(r)
    
    # ====================
    # Exp4: 518 vs 224 input
    # ====================
    r = run_exp("Exp4_Resolution_224",
                f"--save_dir {base_dir}/exp4_224res --ablation_image_size 224",
                categories, args.dataset)
    r['exp_id'] = 'Exp4'
    results.append(r)
    
    # ====================
    # Exp5: n_layers=2 vs 4 vs 6
    # ====================
    for n in [2, 6]:
        r = run_exp(f"Exp5_nLayers_{n}",
                    f"--save_dir {base_dir}/exp5_n{n} --ablation_n_layers {n}",
                    categories, args.dataset)
        r['exp_id'] = 'Exp5'
        results.append(r)
    
    # ====================
    # 汇总
    # ====================
    print(f"\n{'='*60}")
    print(f"  消融实验完成！")
    print(f"{'='*60}")
    for r in results:
        status = '✅' if r['status'] == 'OK' else '❌'
        print(f"  {status} {r['name']}  ({r['time']:.0f}s)")
    
    # 保存JSON
    summary_path = base_dir / f"ablation_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump({'timestamp': timestamp, 'dataset': args.dataset, 
                   'categories': categories, 'results': results}, f, indent=2)
    print(f"\n结果保存至: {summary_path}")


if __name__ == '__main__':
    main()
