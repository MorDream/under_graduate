     1|#!/usr/bin/env python3
     2|"""
     3|4.2 消融实验 — 自动运行 + 结果收集脚本
     4|===========================================
     5|论文第四章第二节所需数据：ViT+ReContrast 消融实验
     6|
     7|实验矩阵：
     8|  Baseline: 全部开启（CutPaste + 交叉重建 + 难例挖掘 + DINOv2预训练 + 518分辨率 + n=4层）
     9|  Exp0:    无 CutPaste 增强
    10|  Exp1:    自重建（关交叉重建）
    11|  Exp2:    无难例挖掘
    12|  Exp3:    随机初始化（无DINOv2预训练）
    13|  Exp4:    224分辨率（vs 基准518）
    14|  Exp5:    n=2层 / n=6层（vs 基准n=4）
    15|
    16|用法：
    17|  # 先快速验证（只跑 carpet）
    18|  python 生成实验数据/4.2_消融实验/run_ablation.py --categories carpet --all
    19|
    20|  # MVTec 全部15类
    21|  python 生成实验数据/4.2_消融实验/run_ablation.py --categories all_mvtec --all
    22|
    23|  # 晶圆全品类
    24|  python 生成实验数据/4.2_消融实验/run_ablation.py --dataset wafer --categories all_wafer --all
    25|
    26|输出：
    27|  生成实验数据/4.2_消融实验/output/
    28|  ├── ablation_carpet_20260514_160000.json    # 原始数据
    29|  ├── ablation_carpet_20260514_160000.md      # 论文表格
    30|  └── ablation_carpet_20260514_160000.csv     # Excel可用
    31|"""
    32|
    33|import os
    34|import sys
    35|import json
    36|import time
    37|import subprocess
    38|import argparse
    39|import re
    40|from datetime import datetime
    41|from pathlib import Path
    42|
    43|# ─── 项目根目录（绝对路径）───
    44|ROOT = '/data/coding/under_graduate'
    45|
    46|# ─── 数据集配置 ───
    47|MVTEC_ALL = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
    48|             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
    49|WAFER_ALL = ['BGA 12x4','BGA S5E 16x7','ESSD 12x4','ESSD 12x5',
    50|             'INAND 19x5','MicroSD 20x4','SDSIP 22x3','UBGA 12x5']
    51|
    52|# ─── 实验定义 ───
    53|EXPERIMENTS = [
    54|    # (实验名, 论文用标签, 额外参数, 基准还是对照)
    55|    ('Baseline',  'Baseline（全开）',    '',                         'baseline'),
    56|    ('Exp0',      '无 CutPaste',         '--ablation_no_cutpaste',   'ablated'),
    57|    ('Exp1',      '自重建',             '--ablation_self_recon',     'ablated'),
    58|    ('Exp2',      '无难例挖掘',          '--ablation_no_hard_mining', 'ablated'),
    59|    ('Exp3',      '随机初始化',          '--ablation_no_pretrained',  'ablated'),
    60|    ('Exp4',      '224分辨率',           '--ablation_image_size 224', 'ablated'),
    61|    ('Exp5_n2',   'n=2层',              '--ablation_n_layers 2',     'ablated'),
    62|    ('Exp5_n6',   'n=6层',              '--ablation_n_layers 6',     'ablated'),
    63|]
    64|
    65|# ─── AUROC 解析 ───
    66|def parse_auroc_f1(output: str):
    67|    """从 recontrast_vit_wafer.py 输出中解析 AUROC 和 F1"""
    68|    # 匹配 "AUROC:0.9234" 或 "AUROC: 0.9234"
    69|    auroc_match = re.search(r'AUROC\s*[:=]\s*([\d.]+)', output)
    70|    f1_match = re.search(r'F1\s*[:=]\s*([\d.]+)', output)
    71|    auroc = float(auroc_match.group(1)) if auroc_match else None
    72|    f1 = float(f1_match.group(1)) if f1_match else None
    73|    return auroc, f1
    74|
    75|
    76|def run_single_exp(category, exp_name, extra_args, save_subdir, dataset='mvtec', 
    77|                   base_cmd=f'python {ROOT}/recontrast_vit_wafer.py', eval_interval=200):
    78|    """运行单个品类×单个实验配置，返回 (success, auroc, f1, elapsed)"""
    79|    save_dir = f"{ROOT}/生成实验数据/4.2_消融实验/output/checkpoints/{save_subdir}/{category.replace(' ','_')}"
    80|    wafer_arg = f"--wafer_data_dir {ROOT}/data" if dataset == 'wafer' else ""
    81|    cmd = (f"cd {ROOT} && {base_cmd} --dataset {dataset} --categories {category} "
    82|           f"{wafer_arg} --save_dir {save_dir} "
    83|           f"{extra_args} --eval_interval {eval_interval}")
    84|    
    85|    print(f"\n{'='*70}")
    86|    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {exp_name} | {category}")
    87|    print(f"  CMD: {cmd}")
    88|    print(f"{'='*70}")
    89|    
    90|    start = time.time()
    91|    try:
    92|        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    93|        elapsed = time.time() - start
    94|        stdout = result.stdout
    95|        stderr = result.stderr
    96|        
    97|        # 从最后50行解析AUROC（最终评估结果在最后）
    98|        tail = '\n'.join(stdout.split('\n')[-100:])
    99|        auroc, f1 = parse_auroc_f1(stdout)
   100|        
   101|        if result.returncode != 0:
   102|            print(f"  ❌ 退出码={result.returncode}")
   103|            if stderr:
   104|                print(f"  STDERR(最后5行):")
   105|                for line in stderr.strip().split('\n')[-5:]:
   106|                    print(f"    {line}")
   107|            return False, auroc, f1, elapsed
   108|        
   109|        print(f"  ✅ 完成 ({elapsed:.0f}s) | AUROC={auroc} | F1={f1}")
   110|        return True, auroc, f1, elapsed
   111|        
   112|    except subprocess.TimeoutExpired:
   113|        elapsed = time.time() - start
   114|        print(f"  ⏱ 超时 (>{elapsed:.0f}s)")
   115|        return False, None, None, elapsed
   116|
   117|
   118|def run_all(categories, dataset, experiments, skip_existing=False):
   119|    """跑全部品类×实验的组合"""
   120|    all_results = {}
   121|    
   122|    for cat in categories:
   123|        cat_key = cat.replace(' ', '_')
   124|        all_results[cat_key] = {}
   125|        
   126|        for exp_id, exp_label, exp_args, exp_type in experiments:
   127|            print(f"\n{'#'*70}")
   128|            print(f"# 品类: {cat}  |  实验: {exp_label}")
   129|            print(f"{'#'*70}")
   130|            
   131|            save_subdir = f"{exp_id}"
   132|            ok, auroc, f1, elapsed = run_single_exp(
   133|                cat, exp_label, exp_args, save_subdir, dataset
   134|            )
   135|            
   136|            all_results[cat_key][exp_id] = {
   137|                'label': exp_label,
   138|                'type': exp_type,
   139|                'success': ok,
   140|                'auroc': auroc,
   141|                'f1': f1,
   142|                'time_s': round(elapsed, 0),
   143|            }
   144|            
   145|            # 每跑完一个实验就增量保存
   146|            _save_results(all_results, categories, dataset)
   147|    
   148|    return all_results
   149|
   150|
   151|def _save_results(all_results, categories, dataset):
   152|    """增量保存结果到 JSON / MD / CSV"""
   153|    out_dir = Path(f'{ROOT}/生成实验数据/4.2_消融实验/output')
   154|    out_dir.mkdir(parents=True, exist_ok=True)
   155|    
   156|    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   157|    cat_label = categories[0].replace(' ', '_') if len(categories) == 1 else f'{len(categories)}cats'
   158|    
   159|    # 1. JSON 原始数据
   160|    json_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.json'
   161|    with open(json_path, 'w', encoding='utf-8') as f:
   162|        json.dump({'dataset': dataset, 'categories': [c.replace(' ','_') for c in categories],
   163|                   'timestamp': timestamp, 'results': all_results}, f, indent=2, ensure_ascii=False)
   164|    
   165|    # 2. Markdown 论文表格
   166|    md_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.md'
   167|    _write_markdown_table(all_results, categories, md_path)
   168|    
   169|    # 3. CSV
   170|    csv_path = out_dir / f'ablation_{cat_label}_{dataset}_{timestamp}.csv'
   171|    _write_csv(all_results, categories, csv_path)
   172|    
   173|    print(f"\n📄 结果已保存: {json_path}")
   174|    print(f"📄 论文表格: {md_path}")
   175|
   176|
   177|def _write_markdown_table(all_results, categories, md_path):
   178|    """生成论文用的 Markdown 表格"""
   179|    lines = []
   180|    lines.append(f"# 消融实验结果（表4-X）\n")
   181|    lines.append(f"数据集：{categories[0] if len(categories)==1 else f'{len(categories)}个品类'}")
   182|    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   183|    
   184|    # 表头
   185|    exp_labels = [e[1] for e in EXPERIMENTS]
   186|    exp_ids = [e[0] for e in EXPERIMENTS]
   187|    
   188|    header = "| 品类 | " + " | ".join(exp_labels) + " |"
   189|    sep = "|------|" + "|".join(["--------" for _ in exp_labels]) + "|"
   190|    lines.append(header)
   191|    lines.append(sep)
   192|    
   193|    for cat in categories:
   194|        cat_key = cat.replace(' ', '_')
   195|        vals = []
   196|        for eid in exp_ids:
   197|            r = all_results.get(cat_key, {}).get(eid, {})
   198|            auroc = r.get('auroc')
   199|            if auroc is not None:
   200|                vals.append(f"{auroc:.4f}")
   201|            else:
   202|                vals.append("—")
   203|        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
   204|    
   205|    lines.append("")
   206|    lines.append("> 注：AUROC 越高越好。Baseline 使用全部改进（CutPaste + 交叉重建 + 难例挖掘 + DINOv2 + 518px + n=4）。")
   207|    
   208|    with open(md_path, 'w', encoding='utf-8') as f:
   209|        f.write('\n'.join(lines))
   210|
   211|
   212|def _write_csv(all_results, categories, csv_path):
   213|    """生成 CSV（方便 Excel 打开）"""
   214|    exp_ids = [e[0] for e in EXPERIMENTS]
   215|    lines = ["品类," + ",".join(exp_ids)]
   216|    for cat in categories:
   217|        cat_key = cat.replace(' ', '_')
   218|        vals = []
   219|        for eid in exp_ids:
   220|            r = all_results.get(cat_key, {}).get(eid, {})
   221|            auroc = r.get('auroc')
   222|            vals.append(f"{auroc:.4f}" if auroc is not None else "")
   223|        lines.append(f"{cat}," + ",".join(vals))
   224|    with open(csv_path, 'w', encoding='utf-8-sig') as f:
   225|        f.write('\n'.join(lines))
   226|
   227|
   228|# ─── Main ───
   229|def main():
   230|    parser = argparse.ArgumentParser(description='4.2 消融实验 — 自动运行 + 结果收集')
   231|    parser.add_argument('--dataset', default='mvtec', choices=['mvtec', 'wafer'])
   232|    parser.add_argument('--categories', default='carpet',
   233|                       help='品类名 或 "all_mvtec" / "all_wafer"')
   234|    parser.add_argument('--all', action='store_true', 
   235|                       help='运行全部消融实验（否则只跑Baseline）')
   236|    parser.add_argument('--epochs', type=int, default=200)
   237|    parser.add_argument('--eval_interval', type=int, default=200,
   238|                       help='多少epoch评估一次（默认200=只在最后评估）')
   239|    args = parser.parse_args()
   240|    
   241|    # 解析品类
   242|    if args.categories == 'all_mvtec':
   243|        categories = MVTEC_ALL
   244|    elif args.categories == 'all_wafer':
   245|        categories = WAFER_ALL
   246|    else:
   247|        categories = [c.strip() for c in args.categories.split(',')]
   248|    
   249|    # 选择实验
   250|    if args.all:
   251|        experiments = EXPERIMENTS
   252|    else:
   253|        experiments = EXPERIMENTS[:1]  # 只跑 Baseline
   254|    
   255|    print(f"\n{'#'*70}")
   256|    print(f"# 4.2 消融实验")
   257|    print(f"# 数据集: {args.dataset}")
   258|    print(f"# 品类: {categories}")
   259|    print(f"# 实验数: {len(categories)} × {len(experiments)} = {len(categories)*len(experiments)}")
   260|    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   261|    print(f"{'#'*70}")
   262|    
   263|    results = run_all(categories, args.dataset, experiments)
   264|    
   265|    # 打印汇总
   266|    print(f"\n{'='*70}")
   267|    print(f"  消融实验全部完成！")
   268|    print(f"{'='*70}")
   269|    for cat in categories:
   270|        cat_key = cat.replace(' ', '_')
   271|        print(f"\n  📦 {cat}:")
   272|        for eid, elabel, _, etype in experiments:
   273|            r = results.get(cat_key, {}).get(eid, {})
   274|            status = '✅' if r.get('success') else '❌'
   275|            auroc = f"AUROC={r['auroc']:.4f}" if r.get('auroc') is not None else 'AUROC=—'
   276|            print(f"     {status} {elabel:16s} {auroc}  ({r.get('time_s',0):.0f}s)")
   277|
   278|
   279|if __name__ == '__main__':
   280|    main()
   281|