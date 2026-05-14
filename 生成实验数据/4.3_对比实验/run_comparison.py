     1|#!/usr/bin/env python3
     2|"""
     3|4.3 对比实验 — 多方法对比 + 结果汇总脚本
     4|===========================================
     5|论文第四章第三节所需数据：ViT+ReContrast vs 其他方法
     6|
     7|方法列表：
     8|  🔥 ViT+ReContrast (DINOv2)  — 本文方法
     9|     MoCo v2 (ViT)            — 动量对比基线
    10|     ReContrast (ResNet)       — CNN版ReContrast
    11|     PaDiM                     — 引用论文数据 (Defard et al., ICPR 2021)
    12|     PatchCore                 — 引用论文数据 (Roth et al., NeurIPS 2021)
    13|
    14|用法：
    15|  # MVTec 单类快速测试
    16|  python 生成实验数据/4.3_对比实验/run_comparison.py --categories carpet
    17|
    18|  # MVTec 全部15类
    19|  python 生成实验数据/4.3_对比实验/run_comparison.py --categories all_mvtec
    20|
    21|  # 晶圆全品类
    22|  python 生成实验数据/4.3_对比实验/run_comparison.py --dataset wafer --categories all_wafer
    23|
    24|  # 只跑某几个方法（跳过已跑完的）
    25|  python 生成实验数据/4.3_对比实验/run_comparison.py --categories carpet --methods vit_recontrast,moco
    26|
    27|输出：
    28|  生成实验数据/4.3_对比实验/output/
    29|  ├── comparison_carpet_mvtec_xxx.json
    30|  ├── comparison_carpet_mvtec_xxx.md       ← 论文对比表
    31|  └── comparison_carpet_mvtec_xxx.csv
    32|"""
    33|
    34|import os
    35|import sys
    36|import json
    37|import time
    38|import subprocess
    39|import argparse
    40|import re
    41|from datetime import datetime
    42|from pathlib import Path
    43|
    44|# ─── 项目根目录（绝对路径）───
    45|ROOT = '/data/coding/under_graduate'
    46|
    47|# ─── 数据集配置 ───
    48|MVTEC_ALL = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
    49|             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
    50|WAFER_ALL = ['BGA 12x4','BGA S5E 16x7','ESSD 12x4','ESSD 12x5',
    51|             'INAND 19x5','MicroSD 20x4','SDSIP 22x3','UBGA 12x5']
    52|
    53|# ─── PaDiM / PatchCore 论文参考数据 (MVTec AD, image-level AUROC) ───
    54|# 来源: PaDiM (Defard et al. ICPR 2021), PatchCore (Roth et al. NeurIPS 2021)
    55|PAPER_REFERENCE = {
    56|    'bottle':      {'PaDiM': 0.994, 'PatchCore': 1.000},
    57|    'cable':       {'PaDiM': 0.979, 'PatchCore': 0.987},
    58|    'capsule':     {'PaDiM': 0.927, 'PatchCore': 0.980},
    59|    'carpet':      {'PaDiM': 0.989, 'PatchCore': 0.991},
    60|    'grid':        {'PaDiM': 0.940, 'PatchCore': 0.988},
    61|    'hazelnut':    {'PaDiM': 0.872, 'PatchCore': 1.000},
    62|    'leather':     {'PaDiM': 0.999, 'PatchCore': 1.000},
    63|    'metal_nut':   {'PaDiM': 0.989, 'PatchCore': 1.000},
    64|    'pill':        {'PaDiM': 0.950, 'PatchCore': 0.975},
    65|    'screw':       {'PaDiM': 0.862, 'PatchCore': 0.987},
    66|    'tile':        {'PaDiM': 0.954, 'PatchCore': 0.994},
    67|    'toothbrush':  {'PaDiM': 0.967, 'PatchCore': 0.978},
    68|    'transistor':  {'PaDiM': 0.982, 'PatchCore': 1.000},
    69|    'wood':        {'PaDiM': 0.988, 'PatchCore': 0.993},
    70|    'zipper':      {'PaDiM': 0.955, 'PatchCore': 0.993},
    71|}
    72|
    73|# ─── 方法定义 ───
    74|# (方法ID, 论文用标签, 命令函数)
    75|METHOD_DEFS = {}
    76|
    77|def _make_methods(output_dir):
    78|    """创建方法定义（动态绑定输出目录）"""
    79|    return {
    80|        'vit_recontrast': {
    81|            'label': 'ViT+ReContrast (本文)',
    82|            'cmd': lambda cat: (
    83|                f"cd {ROOT} && python {ROOT}/recontrast_vit_wafer.py --dataset {_ds} "
    84|                f"--categories {cat} --epochs 200 --eval_interval 200 "
    85|                f"--save_dir {output_dir}/vit_recontrast/{cat.replace(' ','_')}"
    86|            ),
    87|        },
    88|        'moco': {
    89|            'label': 'MoCo v2 (ViT)',
    90|            'cmd': lambda cat: (
    91|                f"cd {ROOT} && python -m wafer_defect_detection.train --mode train "
    92|                f"--dataset {_ds} "
    93|                f"--{_cat_arg} {cat} --epochs 200 --batch_size 32 "
    94|                f"--use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere "
    95|                f"--save_dir {output_dir}/moco --eval_interval 200"
    96|            ),
    97|        },
    98|        'recontrast_resnet': {
    99|            'label': 'ReContrast (ResNet)',
   100|            'cmd': lambda cat: (
   101|                f"cd {ROOT} && python {ROOT}/recontrast_wafer.py --dataset {_ds} "
   102|                f"--categories {cat} --epochs 200 --eval_interval 200 "
   103|                f"--save_dir {output_dir}/recontrast_resnet/{cat.replace(' ','_')}"
   104|            ),
   105|        },
   106|    }
   107|
   108|
   109|# 全局变量（由 run_all 设置）
   110|_ds = 'mvtec'
   111|_cat_arg = 'mvtec_category'
   112|
   113|
   114|def parse_auroc_f1(output: str):
   115|    """从训练脚本输出中解析 AUROC 和 F1"""
   116|    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
   117|    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
   118|    auroc = float(auroc_match.group(1)) if auroc_match else None
   119|    f1 = float(f1_match.group(1)) if f1_match else None
   120|    return auroc, f1
   121|
   122|
   123|def run_method(method_id, method_info, category, dataset):
   124|    """运行单个方法×品类，返回 (success, auroc, f1, elapsed)"""
   125|    global _ds, _cat_arg
   126|    _ds = dataset
   127|    _cat_arg = 'mvtec_category' if dataset == 'mvtec' else 'wafer_category'
   128|    
   129|    cmd = method_info['cmd'](category)
   130|    
   131|    print(f"\n{'='*70}")
   132|    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {method_info['label']} | {category}")
   133|    print(f"  CMD: {cmd}")
   134|    print(f"{'='*70}")
   135|    
   136|    start = time.time()
   137|    try:
   138|        # 用 Popen 实时打印 + 捕获
   139|        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
   140|                                stderr=subprocess.STDOUT, text=True,
   141|                                bufsize=1, universal_newlines=True)
   142|        stdout_lines = []
   143|        for line in proc.stdout:
   144|            print(f"    {line.rstrip()}")
   145|            stdout_lines.append(line)
   146|        proc.wait(timeout=7200)
   147|        elapsed = time.time() - start
   148|        
   149|        stdout = ''.join(stdout_lines)
   150|        auroc, f1 = parse_auroc_f1(stdout)
   151|        
   152|        if proc.returncode != 0:
   153|            print(f"  ❌ 退出码={proc.returncode}")
   154|            return False, auroc, f1, elapsed
   155|        
   156|        print(f"  ✅ 完成 ({elapsed:.0f}s) | AUROC={auroc} | F1={f1}")
   157|        return True, auroc, f1, elapsed
   158|        
   159|    except subprocess.TimeoutExpired:
   160|        elapsed = time.time() - start
   161|        proc.kill()
   162|        print(f"  ⏱ 超时 (>{elapsed:.0f}s)")
   163|        return False, None, None, elapsed
   164|
   165|
   166|def run_all(categories, dataset, method_ids, output_dir):
   167|    """跑全部品类×方法的组合"""
   168|    methods = _make_methods(output_dir)
   169|    methods = {k: v for k, v in methods.items() if k in method_ids}
   170|    
   171|    all_results = {}
   172|    
   173|    for cat in categories:
   174|        cat_key = cat.replace(' ', '_')
   175|        all_results[cat_key] = {}
   176|        
   177|        for mid, minfo in methods.items():
   178|            print(f"\n{'#'*70}")
   179|            print(f"# 品类: {cat}  |  方法: {minfo['label']}")
   180|            print(f"{'#'*70}")
   181|            
   182|            ok, auroc, f1, elapsed = run_method(mid, minfo, cat, dataset)
   183|            
   184|            all_results[cat_key][mid] = {
   185|                'label': minfo['label'],
   186|                'success': ok,
   187|                'auroc': auroc,
   188|                'f1': f1,
   189|                'time_s': round(elapsed, 0),
   190|            }
   191|            _save_results(all_results, categories, dataset, output_dir)
   192|    
   193|    # 注入论文参考数据
   194|    if dataset == 'mvtec':
   195|        for cat_key in all_results:
   196|            for ref_method, ref_data in PAPER_REFERENCE.items():
   197|                if cat_key == ref_method or cat_key in [ref_method.replace(' ', '_')]:
   198|                    for ref_name, ref_val in ref_data.items():
   199|                        all_results[cat_key][ref_name] = {
   200|                            'label': ref_name,
   201|                            'auroc': ref_val,
   202|                            'f1': None,
   203|                            'success': True,
   204|                            'time_s': 0,
   205|                            'is_reference': True,
   206|                        }
   207|    
   208|    _save_results(all_results, categories, dataset, output_dir)
   209|    return all_results
   210|
   211|
   212|def _save_results(all_results, categories, dataset, output_dir):
   213|    """增量保存结果"""
   214|    out_dir = Path(output_dir)
   215|    out_dir.mkdir(parents=True, exist_ok=True)
   216|    
   217|    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   218|    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'
   219|    
   220|    json_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.json'
   221|    with open(json_path, 'w', encoding='utf-8') as f:
   222|        json.dump({'dataset': dataset, 'categories': [c.replace(' ','_') for c in categories],
   223|                   'timestamp': timestamp, 'results': all_results}, f, indent=2, ensure_ascii=False)
   224|    
   225|    md_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.md'
   226|    _write_markdown_table(all_results, categories, md_path)
   227|    
   228|    csv_path = out_dir / f'comparison_{cat_label}_{dataset}_{timestamp}.csv'
   229|    _write_csv(all_results, categories, csv_path)
   230|    
   231|    print(f"\n📄 JSON: {json_path}")
   232|    print(f"📄 表格: {md_path}")
   233|
   234|
   235|def _get_method_order(all_results, method_ids):
   236|    """确定表格列顺序：本文方法在第一个，参考数据在最右边"""
   237|    order = []
   238|    for mid in method_ids:
   239|        label = all_results.get(list(all_results.keys())[0], {}).get(mid, {}).get('label', mid)
   240|        order.append((mid, label))
   241|    # 添加参考方法
   242|    ref_methods = set()
   243|    for cat_results in all_results.values():
   244|        for mid, r in cat_results.items():
   245|            if r.get('is_reference'):
   246|                ref_methods.add((mid, r['label']))
   247|    order += sorted(ref_methods, key=lambda x: x[1])
   248|    return order
   249|
   250|
   251|def _write_markdown_table(all_results, categories, md_path):
   252|    """生成论文对比表"""
   253|    method_ids = list(all_results.get(list(all_results.keys())[0], {}).keys())
   254|    order = _get_method_order(all_results, method_ids)
   255|    
   256|    lines = []
   257|    lines.append(f"# 对比实验结果（表4-X）\n")
   258|    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
   259|    lines.append(f"指标：图像级 AUROC")
   260|    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   261|    
   262|    # 表头
   263|    labels = [l for _, l in order]
   264|    header = "| 品类 | " + " | ".join(labels) + " |"
   265|    sep = "|------|" + "|".join(["--------" for _ in labels]) + "|"
   266|    lines.append(header)
   267|    lines.append(sep)
   268|    
   269|    for cat in categories:
   270|        cat_key = cat.replace(' ', '_')
   271|        vals = []
   272|        best_val = -1
   273|        for mid, _ in order:
   274|            r = all_results.get(cat_key, {}).get(mid, {})
   275|            auroc = r.get('auroc')
   276|            if auroc is not None:
   277|                vals.append(f"{auroc:.4f}")
   278|                if not r.get('is_reference') and auroc > best_val:
   279|                    best_val = auroc
   280|            else:
   281|                vals.append("—")
   282|        # 标注最佳
   283|        if best_val > 0:
   284|            vals = [f"**{v}**" if v != '—' and float(v) == best_val else v for v in vals]
   285|        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
   286|    
   287|    # 平均值行
   288|    lines.append(sep)
   289|    avg_vals = []
   290|    for mid, _ in order:
   291|        vals = []
   292|        for cat in categories:
   293|            cat_key = cat.replace(' ', '_')
   294|            r = all_results.get(cat_key, {}).get(mid, {})
   295|            auroc = r.get('auroc')
   296|            if auroc is not None:
   297|                vals.append(auroc)
   298|        if vals:
   299|            avg_vals.append(f"{sum(vals)/len(vals):.4f}")
   300|        else:
   301|            avg_vals.append("—")
   302|    lines.append(f"| **平均** | " + " | ".join(avg_vals) + " |")
   303|    
   304|    lines.append("")
   305|    lines.append("> 注：粗体为该品类最佳结果。PaDiM/PatchCore 数据引用自原论文。")
   306|    lines.append("> ViT+ReContrast 使用 DINOv2 预训练 + CutPaste + 交叉重建 + 难例挖掘。")
   307|    
   308|    with open(md_path, 'w', encoding='utf-8') as f:
   309|        f.write('\n'.join(lines))
   310|
   311|
   312|def _write_csv(all_results, categories, csv_path):
   313|    """生成 CSV"""
   314|    method_ids = list(all_results.get(list(all_results.keys())[0], {}).keys())
   315|    order = _get_method_order(all_results, method_ids)
   316|    labels = [l for _, l in order]
   317|    
   318|    lines = ["品类," + ",".join(labels)]
   319|    for cat in categories:
   320|        cat_key = cat.replace(' ', '_')
   321|        vals = []
   322|        for mid, _ in order:
   323|            r = all_results.get(cat_key, {}).get(mid, {})
   324|            auroc = r.get('auroc')
   325|            vals.append(f"{auroc:.4f}" if auroc is not None else "")
   326|        lines.append(f"{cat}," + ",".join(vals))
   327|    
   328|    with open(csv_path, 'w', encoding='utf-8-sig') as f:
   329|        f.write('\n'.join(lines))
   330|
   331|
   332|# ─── Main ───
   333|def main():
   334|    parser = argparse.ArgumentParser(description='4.3 对比实验 — 多方法对比')
   335|    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
   336|    parser.add_argument('--categories', default='carpet',
   337|                       help='品类名 或 "all_mvtec" / "all_wafer"')
   338|    parser.add_argument('--methods', default='vit_recontrast,moco,recontrast_resnet',
   339|                       help='要跑的方法，逗号分隔: vit_recontrast,moco,recontrast_resnet')
   340|    parser.add_argument('--epochs', type=int, default=200)
   341|    args = parser.parse_args()
   342|    
   343|    # 解析品类
   344|    if args.categories == 'all_mvtec':
   345|        categories = MVTEC_ALL
   346|    elif args.categories == 'all_wafer':
   347|        categories = WAFER_ALL
   348|    else:
   349|        categories = [c.strip() for c in args.categories.split(',')]
   350|    
   351|    method_ids = [m.strip() for m in args.methods.split(',')]
   352|    output_dir = f'{ROOT}/生成实验数据/4.3_对比实验/output'
   353|    
   354|    print(f"\n{'#'*70}")
   355|    print(f"# 4.3 对比实验")
   356|    print(f"# 数据集: {args.dataset}")
   357|    print(f"# 品类: {categories}")
   358|    print(f"# 方法: {method_ids}")
   359|    print(f"# 组合数: {len(categories)} × {len(method_ids)} = {len(categories)*len(method_ids)}")
   360|    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   361|    print(f"{'#'*70}")
   362|    
   363|    results = run_all(categories, args.dataset, method_ids, output_dir)
   364|    
   365|    # 汇总打印
   366|    print(f"\n{'='*70}")
   367|    print(f"  对比实验全部完成！")
   368|    print(f"{'='*70}")
   369|    for cat in categories:
   370|        cat_key = cat.replace(' ', '_')
   371|        print(f"\n  📦 {cat}:")
   372|        for mid in results.get(cat_key, {}):
   373|            r = results[cat_key][mid]
   374|            ref_mark = '📖' if r.get('is_reference') else ('✅' if r.get('success') else '❌')
   375|            auroc = f"AUROC={r['auroc']:.4f}" if r.get('auroc') is not None else 'AUROC=—'
   376|            print(f"     {ref_mark} {r['label']:24s} {auroc}")
   377|
   378|
   379|if __name__ == '__main__':
   380|    main()
   381|