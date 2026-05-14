#!/usr/bin/env python3
"""
4.3 对比实验 — 多方法对比 + 结果汇总脚本
===========================================
论文第四章第三节所需数据：ViT+ReContrast vs 其他方法

方法列表：
  🔥 ViT+ReContrast (DINOv2)  — 本文方法
     MoCo v2 (ViT)            — 动量对比基线
     ReContrast (ResNet)       — CNN版ReContrast
     PaDiM                     — 引用论文数据 (Defard et al., ICPR 2021)
     PatchCore                 — 引用论文数据 (Roth et al., NeurIPS 2021)

用法：
  python 生成实验数据/4.3_对比实验/run_comparison.py --categories carpet
  python 生成实验数据/4.3_对比实验/run_comparison.py --categories all_mvtec
  python 生成实验数据/4.3_对比实验/run_comparison.py --dataset wafer --categories all_wafer

输出：
  生成实验数据/4.3_对比实验/output/
  ├── comparison_carpet_mvtec_xxx.json / .md / .csv
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
WAFER_ALL = ['BGA 12x4','BGA S5E 16x7','ESSD 12x4','ESSD 12x5',
             'INAND 19x5','MicroSD 20x4','SDSIP 22x3','UBGA 12x5']

PAPER_REFERENCE = {
    'bottle':      {'PaDiM': 0.994, 'PatchCore': 1.000},
    'cable':       {'PaDiM': 0.979, 'PatchCore': 0.987},
    'capsule':     {'PaDiM': 0.927, 'PatchCore': 0.980},
    'carpet':      {'PaDiM': 0.989, 'PatchCore': 0.991},
    'grid':        {'PaDiM': 0.940, 'PatchCore': 0.988},
    'hazelnut':    {'PaDiM': 0.872, 'PatchCore': 1.000},
    'leather':     {'PaDiM': 0.999, 'PatchCore': 1.000},
    'metal_nut':   {'PaDiM': 0.989, 'PatchCore': 1.000},
    'pill':        {'PaDiM': 0.950, 'PatchCore': 0.975},
    'screw':       {'PaDiM': 0.862, 'PatchCore': 0.987},
    'tile':        {'PaDiM': 0.954, 'PatchCore': 0.994},
    'toothbrush':  {'PaDiM': 0.967, 'PatchCore': 0.978},
    'transistor':  {'PaDiM': 0.982, 'PatchCore': 1.000},
    'wood':        {'PaDiM': 0.988, 'PatchCore': 0.993},
    'zipper':      {'PaDiM': 0.955, 'PatchCore': 0.993},
}

_ds = 'mvtec'
_cat_arg = 'mvtec_category'


def _make_methods(output_dir):
    return {
        'vit_recontrast': {
            'label': 'ViT+ReContrast (本文)',
            'cmd': lambda cat, view='': (
                f"cd {ROOT} && python {ROOT}/recontrast_vit_wafer.py "
                f"--dataset {_ds} "
                f"{'--wafer_category \"' + cat + '\" --wafer_view ' + view if _ds == 'wafer' else '--categories ' + cat} "
                f"--eval_interval 200 "
                f"--save_dir {output_dir}/vit_recontrast/{cat.replace(' ','_')}"
            ),
        },
        'moco': {
            'label': 'MoCo v2 (ViT)',
            'cmd': lambda cat, view='': (
                f"cd {ROOT} && python -m wafer_defect_detection.train --mode train "
                f"--dataset {_ds} --{_cat_arg} \"{cat}\" "
                f"{'--wafer_view ' + view if _ds == 'wafer' else ''} "
                f"--epochs 200 --batch_size 32 "
                f"--use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere "
                f"--save_dir {output_dir}/moco --eval_interval 200"
            ),
        },
        'recontrast_resnet': {
            'label': 'ReContrast (ResNet)',
            'cmd': lambda cat, view='': (
                f"cd {ROOT} && python {ROOT}/recontrast_wafer.py "
                f"--dataset {_ds} "
                f"{'--wafer_category \"' + cat + '\" --wafer_view ' + view if _ds == 'wafer' else '--categories ' + cat} "
                f"--eval_interval 200 "
                f"--save_dir {output_dir}/recontrast_resnet/{cat.replace(' ','_')}"
            ),
        },
    }


def parse_auroc_f1(output):
    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
    auroc = float(auroc_match.group(1)) if auroc_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    return auroc, f1


def run_method(method_id, method_info, category, dataset, wafer_view=None):
    global _ds, _cat_arg
    _ds = dataset
    _cat_arg = 'mvtec_category' if dataset == 'mvtec' else 'wafer_category'

    view_suffix = f'_{wafer_view}' if wafer_view else ''
    cmd = method_info['cmd'](category, wafer_view or '')

    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {method_info['label']} | {category}")
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


def run_all(categories, dataset, method_ids, output_dir):
    methods = _make_methods(output_dir)
    methods = {k: v for k, v in methods.items() if k in method_ids}

    # For wafer, each category has UP and DOWN views
    wafer_views = ['UP', 'DOWN'] if dataset == 'wafer' else [None]

    all_results = {}

    for cat in categories:
        cat_key = cat.replace(' ', '_')
        all_results[cat_key] = {}

        for mid, minfo in methods.items():
            view_aurocs = []
            view_f1s = []

            for view in wafer_views:
                view_label = f' {view}' if view else ''
                print(f"\n{'#'*70}")
                print(f"# 品类: {cat}{view_label}  |  方法: {minfo['label']}")
                print(f"{'#'*70}")

                ok, auroc, f1, elapsed = run_method(mid, minfo, cat, dataset, wafer_view=view)

                if auroc is not None:
                    view_aurocs.append(auroc)
                if f1 is not None:
                    view_f1s.append(f1)

            # Average across views for wafer
            avg_auroc = sum(view_aurocs) / len(view_aurocs) if view_aurocs else None
            avg_f1 = sum(view_f1s) / len(view_f1s) if view_f1s else None

            all_results[cat_key][mid] = {
                'label': minfo['label'],
                'success': ok if len(wafer_views) == 1 else (len(view_aurocs) > 0),
                'auroc': avg_auroc,
                'f1': avg_f1,
                'time_s': round(elapsed, 0),
            }
            if len(wafer_views) > 1:
                all_results[cat_key][mid]['view_aurocs'] = dict(zip(wafer_views, view_aurocs))
            _save_results(all_results, categories, dataset, output_dir)

    if dataset == 'mvtec':
        for cat_key in all_results:
            for ref_method, ref_data in PAPER_REFERENCE.items():
                if cat_key == ref_method or cat_key in [ref_method.replace(' ', '_')]:
                    for ref_name, ref_val in ref_data.items():
                        all_results[cat_key][ref_name] = {
                            'label': ref_name,
                            'auroc': ref_val,
                            'f1': None,
                            'success': True,
                            'time_s': 0,
                            'is_reference': True,
                        }

    _save_results(all_results, categories, dataset, output_dir)
    return all_results


def _save_results(all_results, categories, dataset, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'

    json_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'categories': [c.replace(' ','_') for c in categories],
                   'timestamp': timestamp, 'results': all_results}, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.md'
    _write_markdown_table(all_results, categories, md_path)

    csv_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.csv'
    _write_csv(all_results, categories, csv_path)

    print(f"\n📄 JSON: {json_path}")
    print(f"📄 表格: {md_path}")


def _get_method_order(all_results, method_ids):
    first_cat = list(all_results.keys())[0] if all_results else None
    order = []
    if first_cat:
        for mid in method_ids:
            label = all_results[first_cat].get(mid, {}).get('label', mid)
            order.append((mid, label))
    ref_methods = set()
    for cat_results in all_results.values():
        for mid, r in cat_results.items():
            if r.get('is_reference'):
                ref_methods.add((mid, r['label']))
    order += sorted(ref_methods, key=lambda x: x[1])
    return order


def _write_markdown_table(all_results, categories, md_path):
    first_cat = list(all_results.keys())[0] if all_results else None
    if not first_cat:
        return
    method_ids = list(all_results[first_cat].keys())
    order = _get_method_order(all_results, method_ids)

    lines = []
    lines.append(f"# 对比实验结果（表4-X）\n")
    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
    lines.append(f"指标：图像级 AUROC")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    labels = [l for _, l in order]
    header = "| 品类 | " + " | ".join(labels) + " |"
    sep = "|------|" + "|".join(["--------" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)

    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        best_val = -1
        for mid, _ in order:
            r = all_results.get(cat_key, {}).get(mid, {})
            auroc = r.get('auroc')
            if auroc is not None:
                vals.append(f"{auroc:.4f}")
                if not r.get('is_reference') and auroc > best_val:
                    best_val = auroc
            else:
                vals.append("—")
        vals_display = []
        for v in vals:
            if v != '—' and best_val > 0 and float(v) == best_val:
                vals_display.append(f"**{v}**")
            else:
                vals_display.append(v)
        lines.append(f"| {cat} | " + " | ".join(vals_display) + " |")

    lines.append(sep)
    avg_vals = []
    for mid, _ in order:
        vals = []
        for cat in categories:
            cat_key = cat.replace(' ', '_')
            r = all_results.get(cat_key, {}).get(mid, {})
            auroc = r.get('auroc')
            if auroc is not None:
                vals.append(auroc)
        avg_vals.append(f"{sum(vals)/len(vals):.4f}" if vals else "—")
    lines.append(f"| **平均** | " + " | ".join(avg_vals) + " |")

    lines.append("")
    lines.append("> 注：PaDiM/PatchCore 数据引用自原论文。ViT+ReContrast 使用 DINOv2 + CutPaste + 交叉重建。")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_csv(all_results, categories, csv_path):
    first_cat = list(all_results.keys())[0] if all_results else None
    if not first_cat:
        return
    method_ids = list(all_results[first_cat].keys())
    order = _get_method_order(all_results, method_ids)
    labels = [l for _, l in order]

    lines = ["品类," + ",".join(labels)]
    for cat in categories:
        cat_key = cat.replace(' ', '_')
        vals = []
        for mid, _ in order:
            r = all_results.get(cat_key, {}).get(mid, {})
            auroc = r.get('auroc')
            vals.append(f"{auroc:.4f}" if auroc is not None else "")
        lines.append(f"{cat}," + ",".join(vals))
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='4.3 对比实验 — 多方法对比')
    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
    parser.add_argument('--categories', default='carpet',
                       help='品类名 或 "all_mvtec" / "all_wafer"')
    parser.add_argument('--methods', default='vit_recontrast,moco,recontrast_resnet',
                       help='要跑的方法，逗号分隔')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    if args.categories == 'all_mvtec':
        categories = MVTEC_ALL
    elif args.categories == 'all_wafer':
        categories = WAFER_ALL
    else:
        categories = [c.strip() for c in args.categories.split(',')]

    method_ids = [m.strip() for m in args.methods.split(',')]
    output_dir = f'{ROOT}/生成实验数据/4.3_对比实验/output'

    print(f"\n{'#'*70}")
    print(f"# 4.3 对比实验")
    print(f"# 数据集: {args.dataset}")
    print(f"# 品类: {categories}")
    print(f"# 方法: {method_ids}")
    print(f"# 组合数: {len(categories)} × {len(method_ids)} = {len(categories)*len(method_ids)}")
    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    results = run_all(categories, args.dataset, method_ids, output_dir)

    print(f"\n{'='*70}")
    print(f"  对比实验全部完成！")
    print(f"{'='*70}")
    for cat in categories:
        cat_key = cat.replace(' ', '_')
        print(f"\n  📦 {cat}:")
        for mid in results.get(cat_key, {}):
            r = results[cat_key][mid]
            ref_mark = '📖' if r.get('is_reference') else ('✅' if r.get('success') else '❌')
            auroc = f"AUROC={r['auroc']:.4f}" if r.get('auroc') is not None else 'AUROC=—'
            print(f"     {ref_mark} {r['label']:24s} {auroc}")


if __name__ == '__main__':
    main()
