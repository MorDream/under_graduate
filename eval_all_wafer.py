"""
全品类晶圆评估脚本
遍历所有品类 × UP/DOWN 视图，逐一评估并输出汇总表
用法: python eval_all_wafer.py
"""

import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from wafer_defect_detection.data.wafer_dataset import get_wafer_categories

# ============ 配置 ============
DATA_DIR = "./data"
SAVE_DIR = "./checkpoints_v3"
SCORE_MODES = ["combined"]  # 可改成 ["combined", "mahal", "memory", "max"]
# ==============================

data_root = Path(DATA_DIR) / "晶圆分类数据集"
categories = get_wafer_categories(str(data_root))
views = ["UP", "DOWN"]

print("=" * 70)
print(f"全品类晶圆评估 - {len(categories)}个品类 × {len(views)}个视图 × {len(SCORE_MODES)}种评分模式")
print(f"品类列表: {categories}")
print("=" * 70)

all_results = {}

for cat in categories:
    for view in views:
        for score_mode in SCORE_MODES:
            model_tag = f"wafer_{cat}_{view}"
            ckpt = Path(SAVE_DIR) / f"best_model_{model_tag}.pth"

            if not ckpt.exists():
                print(f"\n  ⚠️ 模型不存在，跳过: {ckpt.name}")
                continue

            print(f"\n{'─' * 60}")
            print(f"评估: {cat} | {view} | score_mode={score_mode}")
            print(f"{'─' * 60}")

            cmd = [
                sys.executable, "-m", "wafer_defect_detection.train",
                "--mode", "eval",
                "--dataset", "wafer",
                "--data_dir", DATA_DIR,
                "--wafer_category", cat,
                "--wafer_view", view,
                "--save_dir", SAVE_DIR,
                "--score_mode", score_mode,
            ]

            t0 = time.time()
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            r = subprocess.run(cmd, capture_output=True, text=False, cwd=Path.cwd(), env=env)
            elapsed = time.time() - t0

            # 手动用 UTF-8 解码
            stdout = r.stdout.decode("utf-8", errors="replace")
            stderr = r.stderr.decode("utf-8", errors="replace")

            # 打印模型输出（过滤掉关键行）
            for line in stdout.splitlines():
                if any(kw in line for kw in ["评估结果", "AUROC", "Accuracy", "F1 Score", "混淆矩阵",
                                              "TP=", "FN=", "漏检率", "误检率", "阈值策略",
                                              "FNR优先", "当前使用"]):
                    print(f"  {line.strip()}")

            if r.returncode != 0:
                print(f"  ❌ 评估失败 (exit={r.returncode})")
                if stderr:
                    for line in stderr.splitlines()[-5:]:
                        print(f"  STDERR: {line}")
                continue

            # 读取结果JSON
            result_file = Path(SAVE_DIR) / f"results_{model_tag}.json"
            if result_file.exists():
                res = json.loads(result_file.read_text(encoding="utf-8"))
                # 如果是第一个score_mode，初始化条目
                key = f"{cat}_{view}"
                if key not in all_results:
                    all_results[key] = {}
                all_results[key][score_mode] = res
                print(f"  ✅ {elapsed:.1f}s | AUROC={res['auroc']:.4f} Acc={res['accuracy']:.4f} F1={res['f1']:.4f} FNR={res['fnr']:.4f} FPR={res['fpr']:.4f}")

# ============ 汇总表 ============
print("\n" + "=" * 70)
print("📊 全品类晶圆评估汇总")
print("=" * 70)

header = f"{'品类+视图':30s}"
for sm in SCORE_MODES:
    header += f" {'AUROC':>6s}/{sm[:4]:4s} {'Acc':>5s} {'F1':>5s} {'FNR':>5s} {'FPR':>5s}  "
print(header)
print("─" * 70)

for key in sorted(all_results.keys()):
    row = f"{key:30s}"
    for sm in SCORE_MODES:
        if sm in all_results[key]:
            r = all_results[key][sm]
            row += f"  {r['auroc']:.4f}/{sm[:4]} {r['accuracy']:.4f} {r['f1']:.4f} {r['fnr']:.4f} {r['fpr']:.4f}"
        else:
            row += f"  {'—':>20s}"
    print(row)

print("─" * 70)

# 计算平均
print(f"\n{'平均':30s}", end="")
for sm in SCORE_MODES:
    vals = [r[sm] for r in all_results.values() if sm in r]
    if vals:
        avg_auroc = sum(v["auroc"] for v in vals) / len(vals)
        avg_acc = sum(v["accuracy"] for v in vals) / len(vals)
        avg_f1 = sum(v["f1"] for v in vals) / len(vals)
        avg_fnr = sum(v["fnr"] for v in vals) / len(vals)
        avg_fpr = sum(v["fpr"] for v in vals) / len(vals)
        print(f"  {avg_auroc:.4f}/{sm[:4]} {avg_acc:.4f} {avg_f1:.4f} {avg_fnr:.4f} {avg_fpr:.4f}")
    else:
        print(f"  {'—':>20s}", end="")
print()

# 保存汇总
summary_path = Path(SAVE_DIR) / "wafer_eval_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n[INFO] 详细结果已保存: {summary_path}")

# 统计
total = sum(len(modes) for modes in all_results.values())
success = total
print(f"\n{'='*70}")
print(f"评估完成! {len(all_results)}个模型×{len(SCORE_MODES)}种评分 = {total}次评估")
print(f"{'='*70}")
