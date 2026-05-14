#!/usr/bin/env python3
"""
4.2 消融实验 — 自动运行 + 结果收集脚本
===========================================
论文第四章第二节所需数据：ViT+ReContrast 消融实验

实验矩阵：
  Baseline: 全部开启（CutPaste + 交叉重建 + 难例挖掘 + DINOv2预训练 + 518分辨率 + n=4层）
  Exp0:    无 CutPaste 增强
  Exp1:    自重建（关交叉重建）
  Exp2:    无难例挖掘
  Exp3:    随机初始化（无DINOv2预训练）
  Exp4:    224分辨率（vs 基准518）
  Exp5:    n=2层 / n=6层（vs 基准n=4）

用法：
  # 先快速验证（只跑 carpet）
  python 生成实验数据/4.2_消融实验/run_ablation.py --categories carpet --all

  # MVTec 全部15类
  python 生成实验数据/4.2_消融实验/run_ablation.py --categories all_mvtec --all

  # 晶圆全品类
  python 生成实验数据/4.2_消融实验/run_ablation.py --dataset wafer --categories all_wafer --all

输出：
  生成实验数据/4.2_消融实验/output/
  ├── ablation_carpet_20260514_160000.json    # 原始数据
  ├── ablation_carpet_20260514_160000.md      # 论文表格
  └── ablation_carpet_20260514_160000.csv     # Excel可用
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)  # 实时输出到日志
import json
import time
import subprocess
import argparse
import re
from datetime import datetime
from pathlib import Path

# ─── 项目根目录（绝对路径）───
ROOT = '/data/coding/under_graduate'

# ─── 数据集配置 ───
MVTEC_ALL = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
WAFER_ALL = ['BGA 12x4','BGA S5E 16x7','ESSD 12x4','ESSD 12x5',
             'INAND 19x5','MicroSD 20x4','SDSIP 22x3','UBGA 12x5']

# ─── 实验定义 ───
EXPERIMENTS = [
    ('Baseline',  'Baseline（全开）',    '',                         'baseline'),
    ('Exp0',      '无 CutPaste',         '--ablation_no_cutpaste',   'ablated'),
    ('Exp1',      '自重建',             '--ablation_self_recon',     'ablated'),
    ('Exp2',      '无难例挖掘',          '--ablation_no_hard_mining', 'ablated'),
    ('Exp3',      '随机初始化',          '--ablation_no_pretrained',  'ablated'),
    ('Exp4',      '224分辨率',           '--ablation_image_size 224', 'ablated'),
    ('Exp5_n2',   'n=2层',              '--ablation_n_layers 2',     'ablated'),
    ('Exp5_n6',   'n=6层',              '--ablation_n_layers 6',     'ablated'),
]

# ─── AUROC 解析 ───
def parse_auroc_f1(output: str):
    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
    auroc = float(auroc_match.group(1)) if auroc_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    return auroc, f1


def run_single_exp(category, exp_name, extra_args, save_subdir, dataset='mvtec',
                   base_cmd=None, eval_interval=200):
    if base_cmd is None:
        base_cmd = f'python {ROOT}/recontrast_vit_wafer.py'
    save_dir = f"{ROOT}/生成实验数据/4.2_消融实验/output/checkpoints/{save_subdir}/{category.replace(' ','_')}"
    wafer_arg = f"--wafer_data_dir {ROOT}/data" if dataset == 'wafer' else ""
    cmd = (f"cd {ROOT} && {base_cmd} --dataset {dataset} --categories \"{category}\" "
           f"{wafer_arg} --save_dir {save_dir} "
           f"{extra_args} --eval_interval {eval_interval}")

    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {exp_name} | {category}")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}")

    start = time.time()
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, universal_newlines=True)
        stdout_lines = []
        for line in proc.stdout:
            print(f"    {line.rstrip()}")
            stdout_lines.append(line)
        proc.wait(timeout=7200)
        elapsed = time.time() - start

        stdout = ''.join(stdout_lines)
        auroc, f1 = parse_auroc_f1(stdout)

        if proc.returncode != 0:
            print(f"  ❌ 退出码={proc.returncode}")
            return False, auroc, f1, elapsed

        print(f"  ✅ 完成 ({elapsed:.0f}s) | AUROC={auroc} | F1={f1}")
        return True, auroc, f1, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏱ 超时 (>{elapsed:.0f}s)")
        return False, None, None, elapsed


def run_all(categories, dataset, experiments, skip_existing=False):
    all_results = {}

    for cat in categories:
        cat_key = cat.replace(' ', '_')
        all_results[cat_key] = {}

        for exp_id, exp_label, exp_args, exp_type in experiments:
            print(f"\n{'#'*70}")
            print(f"# 品类: {cat}  |  实验: {exp_label}")
            print(f"{'#'*70}")

            save_subdir = f"{exp_id}"
            ok, auroc, f1, elapsed = run_single_exp(
                cat, exp_label, exp_args, save_subdir, dataset
            )

            all_results[cat_key][exp_id] = {
                'label': exp_label,
                'type': exp_type,
                'success': ok,
                'auroc': auroc,
                'f1': f1,
                'time_s': round(elapsed, 0),
            }

            _save_results(all_results, categories, dataset)

    return all_results


def _save_results(all_results, categories, dataset):
    out_dir = Path(f"{ROOT}/生成实验数据/4.2_消融实验/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'

    json_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'categories': [c.replace(' ','_') for c in categories],
                   'timestamp': timestamp, 'results': all_results}, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.md'
    _write_markdown_table(all_results, categories, md_path)

    csv_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.csv'
    _write_csv(all_results, categories, csv_path)

    print(f"\n📄 结果已保存: {json_path}")
    print(f"📄 论文表格: {md_path}")


def _write_markdown_table(all_results, categories, md_path):
    lines = []
    lines.append(f"# 消融实验结果（表4-X）\n")
    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    exp_labels = [e[1] for e in EXPERIMENTS]
    exp_ids = [e[0] for e in EXPERIMENTS]

    header = "| 品类 | " + " | ".join(exp_labels) + " |"
    sep = "|------|" + "|".join(["--------" for _ in exp_labels]) + "|"
    lines.append(header)
    lines.append(sep)

    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        for eid in exp_ids:
            r = all_results.get(cat_key, {}).get(eid, {})
            auroc = r.get('auroc')
            if auroc is not None:
                vals.append(f"{auroc:.4f}")
            else:
                vals.append("—")
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("> 注：AUROC 越高越好。Baseline 使用全部改进（CutPaste + 交叉重建 + 难例挖掘 + DINOv2 + 518px + n=4）。")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_csv(all_results, categories, csv_path):
    exp_ids = [e[0] for e in EXPERIMENTS]
    lines = ["品类," + ",".join(exp_ids)]
    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        for eid in exp_ids:
            r = all_results.get(cat_key, {}).get(eid, {})
            auroc = r.get('auroc')
            vals.append(f"{auroc:.4f}" if auroc is not None else "")
        lines.append(f"{cat}," + ",".join(vals))
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))


# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description='4.2 消融实验 — 自动运行 + 结果收集')
    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
    parser.add_argument('--categories', default='carpet',
                       help='品类名 或 "all_mvtec" / "all_wafer"')
    parser.add_argument('--all', action='store_true',
                       help='运行全部消融实验（否则只跑Baseline）')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=200,
                       help='多少epoch评估一次（默认200=只在最后评估）')
    args = parser.parse_args()

    if args.categories == 'all_mvtec':
        categories = MVTEC_ALL
    elif args.categories == 'all_wafer':
        categories = WAFER_ALL
    else:
        categories = [c.strip() for c in args.categories.split(',')]

    if args.all:
        experiments = EXPERIMENTS
    else:
        experiments = EXPERIMENTS[:1]

    print(f"\n{'#'*70}")
    print(f"# 4.2 消融实验")
    print(f"# 数据集: {args.dataset}")
    print(f"# 品类: {categories}")
    print(f"# 实验数: {len(categories)} × {len(experiments)} = {len(categories)*len(experiments)}")
    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    results = run_all(categories, args.dataset, experiments)

    print(f"\n{'='*70}")
    print(f"  消融实验全部完成！")
    print(f"{'='*70}")
    for cat in categories:
        cat_key = cat.replace(' ', '_')
        print(f"\n  📦 {cat}:")
        for eid, elabel, _, etype in experiments:
            r = results.get(cat_key, {}).get(eid, {})
            status = '✅' if r.get('success') else '❌'
            auroc = f"AUROC={r['auroc']:.4f}" if r.get('auroc') is not None else 'AUROC=—'
            print(f"     {status} {elabel:16s} {auroc}  ({r.get('time_s',0):.0f}s)")


if __name__ == '__main__':
    main()
