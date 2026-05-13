"""评估 v3 和 baseline 的对比"""
import json, subprocess, sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wafer_defect_detection.data.wafer_dataset import get_wafer_categories

DATA_DIR = "./data"
data_root = Path(DATA_DIR) / "晶圆分类数据集"
categories = ["ESSD 12x5", "BGA S5E 16x7"]
views = ["UP", "DOWN"]
save_dirs = {
    "改进版(v3)": "./checkpoints_v3",
    "纯MoCo基线": "./checkpoints_v3_baseline",
}
all_results = {}

for label, save_dir in save_dirs.items():
    print(f"\n{'='*70}")
    print(f"📊 评估 {label} (save_dir={save_dir})")
    print(f"{'='*70}")
    for cat in categories:
        for view in views:
            model_tag = f"wafer_{cat}_{view}"
            ckpt = Path(save_dir) / f"best_model_{model_tag}.pth"
            if not ckpt.exists():
                print(f"  ⚠️ 跳过: {ckpt.name} 不存在")
                continue
            print(f"\n  ── {cat} | {view} ──")
            cmd = [
                sys.executable, "-m", "wafer_defect_detection.train",
                "--mode", "eval", "--dataset", "wafer",
                "--data_dir", DATA_DIR,
                "--wafer_category", cat,
                "--wafer_view", view,
                "--save_dir", save_dir,
            ]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            r = subprocess.run(cmd, capture_output=True, text=False, cwd=Path.cwd(), env=env)
            stdout = r.stdout.decode("utf-8", errors="replace")
            stderr = r.stderr.decode("utf-8", errors="replace")
            for line in stdout.splitlines():
                if any(kw in line for kw in ["AUROC", "Accuracy", "F1 Score", "混淆矩阵", "TP=", "FN=", "漏检率", "误检率", "FNR优先"]):
                    print(f"    {line.strip()}")
            if r.returncode != 0 and stderr:
                for line in stderr.splitlines()[-3:]:
                    print(f"    STDERR: {line}")
            result_file = Path(save_dir) / f"results_{model_tag}.json"
            if result_file.exists():
                res = json.loads(result_file.read_text(encoding="utf-8"))
                key = f"{cat}_{view}"
                if key not in all_results:
                    all_results[key] = {}
                all_results[key][label] = res

# === 汇总对比表 ===
print(f"\n\n{'='*80}")
print(f"{'📊 全品类评估汇总对比':^80}")
print(f"{'='*80}")
header = f"{'品类+视图':25s}"
for label in save_dirs:
    header += f"  {'AUROC':>6s}  {'Acc':>5s}  {'F1':>5s}  {'FNR':>5s}  {'FPR':>5s}"
print(header)
print(f"{'─'*80}")
for key in sorted(all_results.keys()):
    row = f"{key:25s}"
    for label in save_dirs:
        if label in all_results[key]:
            r = all_results[key][label]
            row += f"  {r['auroc']:.4f}  {r['accuracy']:.4f}  {r['f1']:.4f}  {r['fnr']:.4f}  {r['fpr']:.4f}"
        else:
            row += f"  {'─':>6s}  {'─':>5s}  {'─':>5s}  {'─':>5s}  {'─':>5s}"
    print(row)
print(f"{'─'*80}")

# 平均
for label in save_dirs:
    vals = [all_results[k][label] for k in all_results if label in all_results[k]]
    if vals:
        avg = lambda k: sum(v[k] for v in vals) / len(vals)
        print(f"\n{label} 平均: AUROC={avg('auroc'):.4f}  Acc={avg('accuracy'):.4f}  F1={avg('f1'):.4f}  FNR={avg('fnr'):.4f}  FPR={avg('fpr'):.4f}")

# 提升幅度
print(f"\n{'─'*80}")
print("📈 提升幅度 (改进版 - 基线)")
print(f"{'─'*80}")
for key in sorted(all_results.keys()):
    if "改进版(v3)" in all_results[key] and "纯MoCo基线" in all_results[key]:
        v3 = all_results[key]["改进版(v3)"]
        base = all_results[key]["纯MoCo基线"]
        delta_auroc = v3['auroc'] - base['auroc']
        delta_f1 = v3['f1'] - base['f1']
        delta_fnr = v3['fnr'] - base['fnr']
        print(f"  {key:25s}  AUROC={delta_auroc:+.4f}  F1={delta_f1:+.4f}  FNR={delta_fnr:+.4f}")
print(f"{'='*80}")
