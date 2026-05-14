     1|#!/usr/bin/env python3
     2|"""
     3|4.4 超参数分析 — 关键超参数对比实验
     4|====================================
     5|论文第四章第四节所需数据：超参数对模型性能的影响
     6|
     7|分析维度：
     8|  1. 学习率:       1e-3, 1e-4, 1e-5
     9|  2. Batch Size:   2, 4, 8 (518分辨率下的显存限制)
    10|  3. 难例阈值 α:   动态调度 vs 固定值(0.5/1.0/2.0)
    11|  4. 训练轮数:     100 vs 200 vs 400 epochs
    12|
    13|用法：
    14|  # 跑全部超参数分析（4个维度，carpet单类）
    15|  python 生成实验数据/4.4_超参数分析/run_hyperparams.py --categories carpet --all
    16|
    17|  # 只跑学习率分析
    18|  python 生成实验数据/4.4_超参数分析/run_hyperparams.py --categories carpet --dims lr
    19|
    20|  # 只跑 batch_size
    21|  python 生成实验数据/4.4_超参数分析/run_hyperparams.py --categories carpet --dims batch_size
    22|
    23|输出：
    24|  生成实验数据/4.4_超参数分析/output/
    25|  ├── hyper_lr_carpet_xxx.json / .md / .csv
    26|  ├── hyper_bs_carpet_xxx.json / .md / .csv
    27|  ├── hyper_alpha_carpet_xxx.json / .md / .csv
    28|  └── hyper_epochs_carpet_xxx.json / .md / .csv
    29|"""
    30|
    31|import os
    32|import sys
    33|import json
    34|import time
    35|import subprocess
    36|import argparse
    37|import re
    38|from datetime import datetime
    39|from pathlib import Path
    40|
    41|# ─── 项目根目录（绝对路径）───
    42|ROOT = '/data/coding/under_graduate'
    43|
    44|# ─── 数据集 ───
    45|MVTEC_ALL = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
    46|             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
    47|
    48|# ─── 超参数搜索空间 ───
    49|HP_DIMS = {
    50|    'lr': {
    51|        'label': '学习率',
    52|        'experiments': [
    53|            ('lr=1e-3', '--lr 1e-3'),
    54|            ('lr=1e-4', '--lr 1e-4'),
    55|            ('lr=5e-5', '--lr 5e-5'),
    56|        ],
    57|    },
    58|    'batch_size': {
    59|        'label': 'Batch Size',
    60|        'experiments': [
    61|            ('bs=2', '--batch_size 2'),
    62|            ('bs=4', '--batch_size 4'),
    63|            ('bs=8', '--batch_size 8'),
    64|        ],
    65|    },
    66|    'alpha': {
    67|        'label': '难例阈值 α',
    68|        'experiments': [
    69|            ('α=动态(默认)', ''),                              # 动态调度 α = min(-3+4t/T, 1.0)
    70|            ('α=0.5', '--ablation_alpha 0.5'),                # 宽松
    71|            ('α=1.0', '--ablation_alpha 1.0'),                # 标准
    72|            ('α=2.0', '--ablation_alpha 2.0'),                # 严格
    73|        ],
    74|    },
    75|    'epochs': {
    76|        'label': '训练轮数',
    77|        'experiments': [
    78|            ('epochs=100', '--epochs 100'),
    79|            ('epochs=200', '--epochs 200'),
    80|            ('epochs=400', '--epochs 400'),
    81|        ],
    82|    },
    83|}
    84|
    85|ALL_DIMS = ['lr', 'batch_size', 'alpha', 'epochs']
    86|
    87|
    88|def parse_auroc_f1(output: str):
    89|    """解析 AUROC 和 F1"""
    90|    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
    91|    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
    92|    auroc = float(auroc_match.group(1)) if auroc_match else None
    93|    f1 = float(f1_match.group(1)) if f1_match else None
    94|    return auroc, f1
    95|
    96|
    97|def run_single_hp(category, dim_name, exp_label, extra_args, output_dir, dataset='mvtec'):
    98|    """运行单个超参数实验"""
    99|    safe_cat = category.replace(' ', '_')
   100|    safe_exp = exp_label.replace('=', '_').replace('(', '').replace(')', '').replace(' ', '_')
   101|    save_dir = f"{output_dir}/checkpoints/{dim_name}/{safe_exp}/{safe_cat}"
   102|    
   103|    base_args = f"--dataset {dataset} --categories {category} --save_dir {save_dir} --eval_interval 200"
   104|    cmd = f"cd {ROOT} && python {ROOT}/recontrast_vit_wafer.py {base_args} {extra_args}"
   105|    
   106|    print(f"\n{'='*70}")
   107|    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {dim_name} | {exp_label} | {category}")
   108|    print(f"  CMD: {cmd}")
   109|    print(f"{'='*70}")
   110|    
   111|    start = time.time()
   112|    try:
   113|        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
   114|                                stderr=subprocess.STDOUT, text=True,
   115|                                bufsize=1, universal_newlines=True)
   116|        stdout_lines = []
   117|        for line in proc.stdout:
   118|            print(f"    {line.rstrip()}")
   119|            stdout_lines.append(line)
   120|        proc.wait(timeout=7200)
   121|        elapsed = time.time() - start
   122|        
   123|        stdout = ''.join(stdout_lines)
   124|        auroc, f1 = parse_auroc_f1(stdout)
   125|        
   126|        if proc.returncode != 0:
   127|            print(f"  ❌ 退出码={proc.returncode}")
   128|            return False, auroc, f1, elapsed
   129|        
   130|        print(f"  ✅ 完成 ({elapsed:.0f}s) | AUROC={auroc} | F1={f1}")
   131|        return True, auroc, f1, elapsed
   132|        
   133|    except subprocess.TimeoutExpired:
   134|        elapsed = time.time() - start
   135|        proc.kill()
   136|        print(f"  ⏱ 超时")
   137|        return False, None, None, elapsed
   138|
   139|
   140|def run_dim(categories, dataset, dim_name, dim_config, output_dir):
   141|    """跑一个超参数维度的全部实验"""
   142|    results = {}
   143|    
   144|    for exp_label, exp_args in dim_config['experiments']:
   145|        print(f"\n{'#'*70}")
   146|        print(f"# 维度: {dim_config['label']}  |  实验: {exp_label}")
   147|        print(f"# 品类: {categories}")
   148|        print(f"{'#'*70}")
   149|        
   150|        for cat in categories:
   151|            cat_key = cat.replace(' ', '_')
   152|            if cat_key not in results:
   153|                results[cat_key] = {}
   154|            
   155|            ok, auroc, f1, elapsed = run_single_hp(cat, dim_name, exp_label, exp_args, output_dir, dataset)
   156|            
   157|            results[cat_key][exp_label] = {
   158|                'success': ok, 'auroc': auroc, 'f1': f1, 'time_s': round(elapsed, 0),
   159|            }
   160|    
   161|    return results
   162|
   163|
   164|def save_dim_results(results, categories, dataset, dim_name, dim_config, output_dir):
   165|    """保存单个维度的结果"""
   166|    out_dir = Path(output_dir)
   167|    out_dir.mkdir(parents=True, exist_ok=True)
   168|    
   169|    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   170|    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'
   171|    prefix = f'hyper_{dim_name}_{cat_label}_{dataset}_{timestamp}'
   172|    
   173|    # JSON
   174|    json_path = out_dir / f'{prefix}.json'
   175|    with open(json_path, 'w', encoding='utf-8') as f:
   176|        json.dump({'dim': dim_name, 'label': dim_config['label'],
   177|                   'dataset': dataset, 'categories': [c.replace(' ', '_') for c in categories],
   178|                   'timestamp': timestamp, 'results': results}, f, indent=2, ensure_ascii=False)
   179|    
   180|    # Markdown
   181|    md_path = out_dir / f'{prefix}.md'
   182|    _write_dim_md(results, categories, dim_config, md_path)
   183|    
   184|    # CSV
   185|    csv_path = out_dir / f'{prefix}.csv'
   186|    _write_dim_csv(results, categories, dim_config, csv_path)
   187|    
   188|    print(f"  📄 {json_path}")
   189|    print(f"  📄 {md_path}")
   190|
   191|
   192|def _write_dim_md(results, categories, dim_config, md_path):
   193|    """生成单个超参数维度的 Markdown 表格"""
   194|    exp_labels = [e[0] for e in dim_config['experiments']]
   195|    
   196|    lines = []
   197|    lines.append(f"# 超参数分析：{dim_config['label']}（表4-X）\n")
   198|    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
   199|    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   200|    
   201|    header = "| 品类 | " + " | ".join(exp_labels) + " | **最佳** |"
   202|    sep = "|------|" + "|".join(["--------" for _ in exp_labels]) + "|--------|"
   203|    lines.append(header)
   204|    lines.append(sep)
   205|    
   206|    for cat in categories:
   207|        cat_key = cat.replace(' ', '_')
   208|        vals = []
   209|        best_val = -1
   210|        for el in exp_labels:
   211|            r = results.get(cat_key, {}).get(el, {})
   212|            auroc = r.get('auroc')
   213|            if auroc is not None:
   214|                vals.append(f"{auroc:.4f}")
   215|                if auroc > best_val:
   216|                    best_val = auroc
   217|            else:
   218|                vals.append("—")
   219|        # 加粗最佳
   220|        vals_display = []
   221|        for v in vals:
   222|            if v != '—' and float(v) == best_val and best_val > 0:
   223|                vals_display.append(f"**{v}**")
   224|            else:
   225|                vals_display.append(v)
   226|        lines.append(f"| {cat} | " + " | ".join(vals_display) + f" | **{best_val:.4f}** |")
   227|    
   228|    lines.append("")
   229|    lines.append(f"> 注：粗体为该品类在 {dim_config['label']} 维度下的最优值。默认参数为 lr=1e-3, bs=4, α=动态, epochs=200。")
   230|    
   231|    with open(md_path, 'w', encoding='utf-8') as f:
   232|        f.write('\n'.join(lines))
   233|
   234|
   235|def _write_dim_csv(results, categories, dim_config, csv_path):
   236|    exp_labels = [e[0] for e in dim_config['experiments']]
   237|    lines = ["品类," + ",".join(exp_labels)]
   238|    for cat in categories:
   239|        cat_key = cat.replace(' ', '_')
   240|        vals = []
   241|        for el in exp_labels:
   242|            r = results.get(cat_key, {}).get(el, {})
   243|            auroc = r.get('auroc')
   244|            vals.append(f"{auroc:.4f}" if auroc is not None else "")
   245|        lines.append(f"{cat}," + ",".join(vals))
   246|    with open(csv_path, 'w', encoding='utf-8-sig') as f:
   247|        f.write('\n'.join(lines))
   248|
   249|
   250|# ─── Main ───
   251|def main():
   252|    parser = argparse.ArgumentParser(description='4.4 超参数分析')
   253|    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
   254|    parser.add_argument('--categories', default='carpet',
   255|                       help='品类名 或 "all_mvtec"')
   256|    parser.add_argument('--dims', default='lr',
   257|                       help='超参数维度: lr, batch_size, alpha, epochs, 或 all')
   258|    parser.add_argument('--all', action='store_true', help='跑全部四个维度')
   259|    args = parser.parse_args()
   260|    
   261|    if args.categories == 'all_mvtec':
   262|        categories = MVTEC_ALL
   263|    else:
   264|        categories = [c.strip() for c in args.categories.split(',')]
   265|    
   266|    if args.all:
   267|        dims = ALL_DIMS
   268|    else:
   269|        dims = [d.strip() for d in args.dims.split(',')]
   270|    
   271|    output_dir = f'{ROOT}/生成实验数据/4.4_超参数分析/output'
   272|    
   273|    print(f"\n{'#'*70}")
   274|    print(f"# 4.4 超参数分析")
   275|    print(f"# 数据集: {args.dataset}  |  品类: {categories}")
   276|    print(f"# 分析维度: {[HP_DIMS[d]['label'] for d in dims]}")
   277|    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   278|    print(f"{'#'*70}")
   279|    
   280|    for dim_name in dims:
   281|        dim_config = HP_DIMS[dim_name]
   282|        n_runs = len(categories) * len(dim_config['experiments'])
   283|        print(f"\n{'─'*70}")
   284|        print(f"  [{dim_config['label']}] {len(dim_config['experiments'])}组 × {len(categories)}品类 = {n_runs} 次运行")
   285|        print(f"{'─'*70}")
   286|        
   287|        results = run_dim(categories, args.dataset, dim_name, dim_config, output_dir)
   288|        save_dim_results(results, categories, args.dataset, dim_name, dim_config, output_dir)
   289|    
   290|    print(f"\n{'='*70}")
   291|    print(f"  超参数分析全部完成！")
   292|    print(f"{'='*70}")
   293|
   294|
   295|if __name__ == '__main__':
   296|    main()
   297|