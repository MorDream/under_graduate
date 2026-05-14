#!/usr/bin/env python3
"""
4.4 超参数分析 — 关键超参数对比实验
====================================
论文第四章第四节：lr / batch_size / alpha / epochs

用法：
  python 生成实验数据/4.4_超参数分析/run_hyperparams.py --categories carpet --all
  python 生成实验数据/4.4_超参数分析/run_hyperparams.py --categories carpet --dims lr

输出：
  生成实验数据/4.4_超参数分析/output/
  ├── hyper_lr_carpet_xxx.json / .md / .csv
  ├── hyper_bs_carpet_xxx.json / .md / .csv
  ├── hyper_alpha_carpet_xxx.json / .md / .csv
  └── hyper_epochs_carpet_xxx.json / .md / .csv
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from datetime import datetime
from pathlib import Path

# ─── 项目根目录（绝对路径）───
ROOT = '/data/coding/under_graduate'

MVTEC_ALL = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

HP_DIMS = {
    'lr': {
        'label': '学习率',
        'experiments': [
            ('lr=1e-3', ''),
            ('lr=1e-4', ''),
            ('lr=5e-5', ''),
        ],
    },
    'batch_size': {
        'label': 'Batch Size',
        'experiments': [
            ('bs=2', ''),
            ('bs=4', ''),
            ('bs=8', ''),
        ],
    },
    'alpha': {
        'label': '难例阈值 α',
        'experiments': [
            ('α=动态(默认)', ''),
            ('α=0.5', ''),
            ('α=1.0', ''),
            ('α=2.0', ''),
        ],
    },
    'epochs': {
        'label': '训练轮数',
        'experiments': [
            ('epochs=100', ''),
            ('epochs=200', ''),
            ('epochs=400', ''),
        ],
    },
}

ALL_DIMS = ['lr', 'batch_size', 'alpha', 'epochs']


def parse_auroc_f1(output):
    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
    auroc = float(auroc_match.group(1)) if auroc_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    return auroc, f1


def run_single_hp(category, dim_name, exp_label, extra_args, output_dir, dataset='mvtec'):
    safe_cat = category.replace(' ', '_')
    safe_exp = exp_label.replace('=', '_').replace('(', '').replace(')', '').replace(' ', '_')
    save_dir = f"{output_dir}/checkpoints/{dim_name}/{safe_exp}/{safe_cat}"

    cmd = (f"cd {ROOT} && python {ROOT}/recontrast_vit_wafer.py "
           f"--dataset {dataset} --categories {category} "
           f"--save_dir {save_dir} --eval_interval 200 "
           f"{extra_args}")

    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {dim_name} | {exp_label} | {category}")
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
        proc.kill()
        print(f"  ⏱ 超时")
        return False, None, None, elapsed


def run_dim(categories, dataset, dim_name, dim_config, output_dir):
    results = {}
    for exp_label, exp_args in dim_config['experiments']:
        print(f"\n{'#'*70}")
        print(f"# 维度: {dim_config['label']}  |  实验: {exp_label}")
        print(f"# 品类: {categories}")
        print(f"{'#'*70}")

        for cat in categories:
            cat_key = cat.replace(' ', '_')
            if cat_key not in results:
                results[cat_key] = {}

            ok, auroc, f1, elapsed = run_single_hp(cat, dim_name, exp_label, exp_args, output_dir, dataset)
            results[cat_key][exp_label] = {
                'success': ok, 'auroc': auroc, 'f1': f1, 'time_s': round(elapsed, 0),
            }
    return results


def save_dim_results(results, categories, dataset, dim_name, dim_config, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'
    prefix = f'hyper_{dim_name}_{cat_label}_{dataset}_{timestamp}'

    json_path = out_dir / f'{prefix}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'dim': dim_name, 'label': dim_config['label'],
                   'dataset': dataset, 'categories': [c.replace(' ', '_') for c in categories],
                   'timestamp': timestamp, 'results': results}, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f'{prefix}.md'
    _write_dim_md(results, categories, dim_config, md_path)

    csv_path = out_dir / f'{prefix}.csv'
    _write_dim_csv(results, categories, dim_config, csv_path)

    print(f"  📄 {json_path}")
    print(f"  📄 {md_path}")


def _write_dim_md(results, categories, dim_config, md_path):
    exp_labels = [e[0] for e in dim_config['experiments']]
    lines = []
    lines.append(f"# 超参数分析：{dim_config['label']}（表4-X）\n")
    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    header = "| 品类 | " + " | ".join(exp_labels) + " | **最佳** |"
    sep = "|------|" + "|".join(["--------" for _ in exp_labels]) + "|--------|"
    lines.append(header)
    lines.append(sep)

    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        best_val = -1
        for el in exp_labels:
            r = results.get(cat_key, {}).get(el, {})
            auroc = r.get('auroc')
            if auroc is not None:
                vals.append(f"{auroc:.4f}")
                if auroc > best_val:
                    best_val = auroc
            else:
                vals.append("—")
        vals_display = []
        for v in vals:
            if v != '—' and float(v) == best_val and best_val > 0:
                vals_display.append(f"**{v}**")
            else:
                vals_display.append(v)
        lines.append(f"| {cat} | " + " | ".join(vals_display) + f" | **{best_val:.4f}** |")

    lines.append("")
    lines.append(f"> 注：粗体为该品类在 {dim_config['label']} 维度下的最优值。")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_dim_csv(results, categories, dim_config, csv_path):
    exp_labels = [e[0] for e in dim_config['experiments']]
    lines = ["品类," + ",".join(exp_labels)]
    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        for el in exp_labels:
            r = results.get(cat_key, {}).get(el, {})
            auroc = r.get('auroc')
            vals.append(f"{auroc:.4f}" if auroc is not None else "")
        lines.append(f"{cat}," + ",".join(vals))
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='4.4 超参数分析')
    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
    parser.add_argument('--categories', default='carpet',
                       help='品类名 或 "all_mvtec"')
    parser.add_argument('--dims', default='lr',
                       help='超参数维度: lr, batch_size, alpha, epochs, 或 all')
    parser.add_argument('--all', action='store_true', help='跑全部四个维度')
    args = parser.parse_args()

    if args.categories == 'all_mvtec':
        categories = MVTEC_ALL
    else:
        categories = [c.strip() for c in args.categories.split(',')]

    if args.all:
        dims = ALL_DIMS
    else:
        dims = [d.strip() for d in args.dims.split(',')]

    output_dir = f'{ROOT}/生成实验数据/4.4_超参数分析/output'

    print(f"\n{'#'*70}")
    print(f"# 4.4 超参数分析")
    print(f"# 数据集: {args.dataset}  |  品类: {categories}")
    print(f"# 分析维度: {[HP_DIMS[d]['label'] for d in dims]}")
    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    for dim_name in dims:
        dim_config = HP_DIMS[dim_name]
        n_runs = len(categories) * len(dim_config['experiments'])
        print(f"\n{'─'*70}")
        print(f"  [{dim_config['label']}] {len(dim_config['experiments'])}组 × {len(categories)}品类 = {n_runs} 次运行")
        print(f"{'─'*70}")

        results = run_dim(categories, args.dataset, dim_name, dim_config, output_dir)
        save_dim_results(results, categories, args.dataset, dim_name, dim_config, output_dir)

    print(f"\n{'='*70}")
    print(f"  超参数分析全部完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
