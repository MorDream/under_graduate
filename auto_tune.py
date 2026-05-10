#!/usr/bin/env python3
"""
自动超参数调优脚本
"""
import subprocess
import json
import os
from datetime import datetime

# 超参数网格
configs = [
    {'lr': 0.0001, 'batch_size': 16, 'epochs': 30, 'name': 'baseline'},
    {'lr': 0.0005, 'batch_size': 16, 'epochs': 30, 'name': 'lr_high'},
    {'lr': 0.0001, 'batch_size': 32, 'epochs': 30, 'name': 'batch_large'},
]

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"实验: {cfg['name']}")
    print(f"lr={cfg['lr']}, bs={cfg['batch_size']}, epochs={cfg['epochs']}")
    print('='*60)
    
    save_dir = f"checkpoints_v3/tune_{cfg['name']}"
    
    # 训练
    train_cmd = [
        'python', 'train_improved_v3.py',
        '--mode', 'train',
        '--dataset', 'wafer',
        '--data_dir', 'data',
        '--epochs', str(cfg['epochs']),
        '--batch_size', str(cfg['batch_size']),
        '--lr', str(cfg['lr']),
        '--use_cutpaste', '--use_feature_generator', '--use_hypersphere',
        '--save_dir', save_dir
    ]
    
    print(f"训练命令: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"训练失败: {result.stderr[-500:]}")
        continue
    
    # 评估
    eval_cmd = [
        'python', 'train_improved_v3.py',
        '--mode', 'eval',
        '--dataset', 'wafer',
        '--data_dir', 'data',
        '--checkpoint', f'{save_dir}/best_model_wafer.pth',
        '--use_cutpaste', '--use_feature_generator', '--use_hypersphere'
    ]
    
    print(f"评估命令: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    # 解析结果
    output = result.stdout
    auroc = f1 = acc = 0
    
    for line in output.split('\n'):
        if 'AUROC:' in line:
            auroc = float(line.split(':')[1].strip())
        elif 'F1 Score:' in line:
            f1 = float(line.split(':')[1].strip())
        elif 'Accuracy:' in line:
            acc = float(line.split(':')[1].strip())
    
    results.append({
        'config': cfg,
        'auroc': auroc,
        'f1': f1,
        'accuracy': acc
    })
    
    print(f"结果: AUROC={auroc:.4f}, F1={f1:.4f}, Accuracy={acc:.4f}")

# 保存结果
with open('checkpoints_v3/tune_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 打印汇总
print(f"\n{'='*60}")
print("调优结果汇总")
print('='*60)
for r in results:
    print(f"{r['config']['name']}: AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}, Acc={r['accuracy']:.4f}")

# 找最佳
best = max(results, key=lambda x: x['auroc'])
print(f"\n最佳配置: {best['config']['name']}")
print(f"  AUROC: {best['auroc']:.4f}")
print(f"  F1: {best['f1']:.4f}")
print(f"  Accuracy: {best['accuracy']:.4f}")
