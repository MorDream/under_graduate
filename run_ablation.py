"""
消融实验 - 逐步添加模块，验证各模块贡献
============================================
实验设计：
  Exp0: ViT + MoCo (纯baseline，无多尺度、无CutPaste、无辅助损失)
  Exp1: + 多尺度特征融合
  Exp2: + CutPaste合成异常增强
  Exp3: + 超球面约束损失 (CFA)
  Exp4: + 特征生成-判别框架 (SimpleNet)
  Exp5: + 记忆库辅助检测 (PatchCore风格) → 完整版

每个实验独立训练+评估，输出AUROC/F1/Accuracy对比表
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "wafer_defect_detection"))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score

from wafer_defect_detection.utils import get_device, set_seed, SemiconductorTransform, EvalTransform
from wafer_defect_detection.data import WaferTrainDataset, WaferEvalDataset
from wafer_defect_detection.models import ViTEncoder, ImprovedMoCo
from wafer_defect_detection.detectors import AnomalyDetector


# ============================================================
# 实验配置
# ============================================================
EXPERIMENTS = [
    {
        'name': 'Exp0_Baseline',
        'desc': 'ViT+MoCo (纯baseline)',
        'use_multiscale': False,
        'use_cutpaste': False,
        'use_hypersphere': False,
        'use_feature_generator': False,
        'use_memory_bank': False,
        'hypersphere_weight': 0,
        'discriminator_weight': 0,
        'generator_weight': 0,
    },
    {
        'name': 'Exp1_Multiscale',
        'desc': '+ 多尺度特征融合',
        'use_multiscale': True,
        'use_cutpaste': False,
        'use_hypersphere': False,
        'use_feature_generator': False,
        'use_memory_bank': False,
        'hypersphere_weight': 0,
        'discriminator_weight': 0,
        'generator_weight': 0,
    },
    {
        'name': 'Exp2_CutPaste',
        'desc': '+ CutPaste合成异常',
        'use_multiscale': True,
        'use_cutpaste': True,
        'use_hypersphere': False,
        'use_feature_generator': False,
        'use_memory_bank': False,
        'hypersphere_weight': 0,
        'discriminator_weight': 0,
        'generator_weight': 0,
    },
    {
        'name': 'Exp3_Hypersphere',
        'desc': '+ 超球面约束(CFA)',
        'use_multiscale': True,
        'use_cutpaste': True,
        'use_hypersphere': True,
        'use_feature_generator': False,
        'use_memory_bank': False,
        'hypersphere_weight': 0.1,
        'discriminator_weight': 0,
        'generator_weight': 0,
    },
    {
        'name': 'Exp4_SimpleNet',
        'desc': '+ 特征生成-判别(SimpleNet)',
        'use_multiscale': True,
        'use_cutpaste': True,
        'use_hypersphere': True,
        'use_feature_generator': True,
        'use_memory_bank': False,
        'hypersphere_weight': 0.1,
        'discriminator_weight': 0.05,
        'generator_weight': 0.05,
    },
    {
        'name': 'Exp5_MemoryBank',
        'desc': '+ 记忆库(PatchCore) = 完整版',
        'use_multiscale': True,
        'use_cutpaste': True,
        'use_hypersphere': True,
        'use_feature_generator': True,
        'use_memory_bank': True,
        'hypersphere_weight': 0.1,
        'discriminator_weight': 0.05,
        'generator_weight': 0.05,
    },
]


def run_single_experiment(exp_config, args):
    """运行单个实验：训练 + 评估"""
    name = exp_config['name']
    desc = exp_config['desc']
    print(f"\n{'='*70}")
    print(f"  {name}: {desc}")
    print(f"{'='*70}")
    
    device = get_device()
    set_seed(args.seed)
    
    save_dir = Path(args.save_dir) / name
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # ---- 数据集 ----
    data_root = Path(args.data_dir) / "晶圆分类数据集"
    transform = SemiconductorTransform(
        img_size=args.img_size,
        use_cutpaste=exp_config['use_cutpaste'],
        cutpaste_prob=args.cutpaste_prob,
    )
    eval_transform = EvalTransform(img_size=args.img_size)
    
    # 训练集（只用正常样本）
    train_dataset = WaferTrainDataset(data_root, transform=transform, val_ratio=0.2, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type == 'cuda')
    )
    
    # 评估集
    eval_train_dataset = WaferEvalDataset(data_root, transform=eval_transform, val_ratio=0.2, split='train')
    eval_val_dataset = WaferEvalDataset(data_root, transform=eval_transform, val_ratio=0.2, split='val')
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    eval_val_loader = DataLoader(eval_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # ---- 模型 ----
    model = ImprovedMoCo(
        embed_dim=args.embed_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        img_size=args.img_size,
        use_multiscale=exp_config['use_multiscale'],
        use_feature_generator=exp_config['use_feature_generator'],
        use_hypersphere=exp_config['use_hypersphere'],
    ).to(device)
    
    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ---- 训练 ----
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for img_q, img_k in train_loader:
            img_q = img_q.to(device)
            img_k = img_k.to(device)
            
            losses, _ = model(img_q, img_k, epoch=epoch, total_epochs=args.epochs)
            
            loss = losses['contrastive'] + \
                   exp_config['hypersphere_weight'] * losses['hypersphere'] + \
                   exp_config['discriminator_weight'] * losses['discriminator'] + \
                   exp_config['generator_weight'] * losses['generator']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} (best: {best_loss:.4f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_q_state_dict': model.encoder_q.state_dict(),
                'projector_q_state_dict': model.projector_q.state_dict(),
                'loss': best_loss,
                'args': vars(args),
                'exp_config': exp_config,
            }, save_dir / "best_model.pth")
    
    train_time = time.time() - start_time
    
    # ---- 评估 ----
    encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
    checkpoint = torch.load(save_dir / "best_model.pth", map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_q_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    use_multiscale = exp_config['use_multiscale']
    
    detector = AnomalyDetector(
        device=device,
        n_components=None,
        use_hypersphere=exp_config['use_hypersphere'],
        use_memory_bank=exp_config.get('use_memory_bank', False),
        memory_ratio=0.1,
        min_pca_components=32,
        pca_variance=0.995,
        score_mode='combined' if exp_config.get('use_memory_bank', False) else 'mahal',
    )
    detector.fit(encoder, eval_train_loader, use_multiscale=use_multiscale)
    
    scores, labels, paths = detector.score(encoder, eval_val_loader, use_multiscale=use_multiscale)
    
    # 计算指标
    auroc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    preds = (scores > best_threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fnr = fn / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    
    result = {
        'name': name,
        'desc': desc,
        'auroc': float(auroc),
        'f1': float(best_f1),
        'accuracy': float(accuracy),
        'fnr': float(fnr),
        'fpr': float(fpr),
        'threshold': float(best_threshold),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'train_time_sec': round(train_time, 1),
        'best_loss': float(best_loss),
    }
    
    # 保存结果
    with open(save_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n--- {name} 结果 ---")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  F1:    {best_f1:.4f}")
    print(f"  Acc:   {accuracy:.4f}")
    print(f"  FNR:   {fnr:.4f}  FPR: {fpr:.4f}")
    print(f"  训练时间: {train_time:.0f}s")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./ablation_results')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--queue_size', type=int, default=1024)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cutpaste_prob', type=float, default=0.3)
    # 可选：只跑某几个实验
    parser.add_argument('--exp_range', type=str, default=None,
                       help='只跑指定范围，如 "0-2" 或 "3,5"')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  消融实验 - 逐步验证各模块贡献")
    print("=" * 70)
    
    # 确定要跑的实验
    exp_indices = list(range(len(EXPERIMENTS)))
    if args.exp_range:
        if '-' in args.exp_range:
            start, end = args.exp_range.split('-')
            exp_indices = list(range(int(start), int(end) + 1))
        else:
            exp_indices = [int(x) for x in args.exp_range.split(',')]
    
    all_results = []
    
    for idx in exp_indices:
        exp = EXPERIMENTS[idx]
        print(f"\n>>> 实验 [{idx}/{len(EXPERIMENTS)-1}]: {exp['name']}")
        result = run_single_experiment(exp, args)
        all_results.append(result)
    
    # ============================================================
    # 汇总对比表
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  消融实验汇总")
    print(f"{'='*80}")
    print(f"{'实验':<25} {'描述':<25} {'AUROC':>8} {'F1':>8} {'Acc':>8} {'FNR':>8} {'FPR':>8}")
    print(f"{'-'*80}")
    
    for r in all_results:
        print(f"{r['name']:<25} {r['desc']:<25} "
              f"{r['auroc']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>8.4f} "
              f"{r['fnr']:>8.4f} {r['fpr']:>8.4f}")
    
    # 计算每个模块的增量贡献
    print(f"\n{'='*80}")
    print(f"  各模块增量贡献 (相比前一实验的提升)")
    print(f"{'='*80}")
    for i in range(1, len(all_results)):
        prev = all_results[i-1]
        curr = all_results[i]
        delta_auroc = curr['auroc'] - prev['auroc']
        delta_f1 = curr['f1'] - prev['f1']
        sign_a = '+' if delta_auroc >= 0 else ''
        sign_f = '+' if delta_f1 >= 0 else ''
        print(f"  {curr['desc']:<30} AUROC: {sign_a}{delta_auroc:.4f}  F1: {sign_f}{delta_f1:.4f}")
    
    # 保存汇总
    save_dir = Path(args.save_dir)
    with open(save_dir / "ablation_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 汇总结果已保存到: {save_dir / 'ablation_summary.json'}")


if __name__ == '__main__':
    main()
