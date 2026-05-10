"""
半导体晶圆无监督缺陷检测 - 改进版V3 (模块化版本)
=====================================
基于ViT+MoCo主体架构，融合前沿方法进行改进：
- SimpleNet: 特征生成-判别框架（辅助损失）
- CFA: 超球面约束损失
- CutPaste: 合成异常增强
- 难负样本挖掘
- 支持MVTec AD开源数据集验证

主体架构保持不变：ViT Encoder + MoCo v2框架
运行方式：python train_improved_v3.py
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from wafer_defect_detection.utils import get_device, set_seed, SemiconductorTransform, EvalTransform
from wafer_defect_detection.data import (
    WaferDataset, WaferTrainDataset, WaferEvalDataset,
    MVTecDataset, MVTecTrainDataset, MVTecEvalDataset,
    get_mvtec_categories
)
from wafer_defect_detection.models import ViTEncoder, ImprovedMoCo
from wafer_defect_detection.detectors import AnomalyDetector


# ============================================================
# 7. 训练流程
# ============================================================
def train(args):
    """训练模型"""
    device = get_device()
    set_seed(args.seed)

    # 数据集
    if args.dataset == 'wafer':
        data_root = Path(args.data_dir) / "数据集" / "数据集"
        transform = SemiconductorTransform(img_size=args.img_size, 
                                           use_cutpaste=args.use_cutpaste,
                                           cutpaste_prob=args.cutpaste_prob)
        # 使用 WaferTrainDataset：split='train' 仅用正常样本进行对比学习
        dataset = WaferTrainDataset(data_root, transform=transform,
                                    val_ratio=args.val_ratio, split='train')
    elif args.dataset == 'mvtec':
        data_root = Path(args.mvtec_dir)
        transform = SemiconductorTransform(img_size=args.img_size,
                                           use_cutpaste=args.use_cutpaste,
                                           cutpaste_prob=args.cutpaste_prob)
        # 使用新的MVTecTrainDataset，返回两个视图
        dataset = MVTecTrainDataset(data_root, args.mvtec_category, transform=transform)
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type in ['cuda', 'xpu'])
    )

    # 模型
    model = ImprovedMoCo(
        embed_dim=args.embed_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        img_size=args.img_size,
        use_multiscale=args.use_multiscale,
        use_feature_generator=args.use_feature_generator,
        use_hypersphere=args.use_hypersphere,
    ).to(device)

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print(f"\n{'='*60}")
    print(f"开始训练 | 设备: {device} | 数据集: {args.dataset}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"改进: CutPaste={args.use_cutpaste}, 特征生成器={args.use_feature_generator}, 超球面={args.use_hypersphere}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loss_components = {'contrastive': 0, 'hypersphere': 0, 'discriminator': 0, 'generator': 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img_q, img_k in pbar:
            img_q = img_q.to(device)
            img_k = img_k.to(device)

            losses, _ = model(img_q, img_k, epoch=epoch, total_epochs=args.epochs)
            
            # 组合损失
            loss = losses['contrastive'] + \
                   args.hypersphere_weight * losses['hypersphere'] + \
                   args.discriminator_weight * losses['discriminator'] + \
                   args.generator_weight * losses['generator']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += losses[k].item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'temp': f"{model.temperature:.3f}"
            })

        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} "
              f"(C:{loss_components['contrastive']:.3f} H:{loss_components['hypersphere']:.3f} "
              f"D:{loss_components['discriminator']:.3f} G:{loss_components['generator']:.3f}) "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_q_state_dict': model.encoder_q.state_dict(),
                'projector_q_state_dict': model.projector_q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, save_dir / f"best_model_{args.dataset}.pth")
            print(f"  -> 保存最佳模型")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'encoder_q_state_dict': model.encoder_q.state_dict(),
        'projector_q_state_dict': model.projector_q.state_dict(),
        'loss': avg_loss,
        'args': vars(args),
    }, save_dir / f"final_model_{args.dataset}.pth")
    
    print(f"\n训练完成! 最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")

    return model


# ============================================================
# 8. 评估流程 - 支持MVTec AD
# ============================================================
def evaluate(args):
    """评估模型"""
    device = get_device()

    # 加载训练数据（用于fit异常检测器）
    eval_transform = EvalTransform(img_size=args.img_size)
    if args.dataset == 'wafer':
        data_root = Path(args.data_dir) / "数据集" / "数据集"

        if args.val_ratio > 0:
            # 训练集(1-val_ratio)：用来fit异常检测器（仅正常样本）
            train_dataset = WaferEvalDataset(
                data_root, transform=eval_transform,
                val_ratio=args.val_ratio, split='train'
            )
            # 验证集(val_ratio)：正常+缺陷，用来评估
            eval_dataset = WaferEvalDataset(
                data_root, transform=eval_transform,
                val_ratio=args.val_ratio, split='val'
            )
            print(f"[INFO] 验证集划分: val_ratio={args.val_ratio}")
        else:
            # 兼容旧接口：全部数据（用 WaferEvalDataset 确保返回三元组）
            eval_dataset = WaferEvalDataset(data_root, transform=eval_transform)
            train_dataset = WaferEvalDataset(data_root, transform=eval_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )
    elif args.dataset == 'mvtec':
        data_root = Path(args.mvtec_dir)
        eval_dataset = MVTecEvalDataset(data_root, args.mvtec_category, 
                                    transform=eval_transform, phase='test')
        # MVTec需要从训练集的good样本fit
        train_dataset = MVTecEvalDataset(data_root, args.mvtec_category,
                                     transform=eval_transform, phase='train')
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # 加载模型
    encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_q_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    use_multiscale = checkpoint.get('args', {}).get('use_multiscale', True)
    print(f"[INFO] 加载模型: {args.checkpoint}")

    # 异常检测 - 使用训练集fit，测试集score
    detector = AnomalyDetector(
        device=device, 
        n_components=args.pca_components,
        use_hypersphere=args.use_hypersphere,
        use_memory_bank=True,
        memory_ratio=0.05,
        min_pca_components=32,
        pca_variance=0.995,
        score_mode=args.score_mode,
    )
    print(f"[INFO] 使用训练集拟合异常检测器...")
    detector.fit(encoder, train_loader, use_multiscale=use_multiscale)

    print(f"[INFO] 在测试集上评估...")
    scores, labels, paths = detector.score(encoder, eval_loader, use_multiscale=use_multiscale)

    # 计算指标
    auroc = roc_auc_score(labels, scores)
    
    # === 策略1: F1最优阈值（传统） ===
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_thresh_idx]

    # === 策略2: FNR优先阈值（宁可误杀不漏杀） ===
    best_threshold_fnr = detector.find_threshold_fnr_priority(scores, labels, target_fnr=0.0)

    # 使用FNR优先阈值作为最终结果
    best_threshold = best_threshold_fnr

    # 计算准确率
    preds = (scores > best_threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    
    # 混淆矩阵
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    print(f"\n{'='*60}")
    print(f"评估结果 - {args.dataset} {'(' + args.mvtec_category + ')' if args.dataset == 'mvtec' else ''}")
    print(f"{'='*60}")
    print(f"AUROC: {auroc:.4f}")
    print(f"\n--- 阈值策略对比 ---")
    print(f"F1最优阈值: {best_threshold_f1:.4f} (F1={best_f1:.4f})")
    print(f"FNR优先阈值: {best_threshold_fnr:.4f} (目标0漏检)")
    print(f"\n当前使用: FNR优先阈值 = {best_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP={tp} FP={fp}")
    print(f"  FN={fn} TN={tn}")
    print(f"  漏检率(FNR): {fn/(tp+fn+1e-8):.4f}")
    print(f"  误检率(FPR): {fp/(fp+tn+1e-8):.4f}")
    print(f"{'='*60}")

    # 保存结果
    results = {
        'dataset': args.dataset,
        'category': args.mvtec_category if args.dataset == 'mvtec' else None,
        'auroc': float(auroc),
        'f1': float(best_f1),
        'accuracy': float(accuracy),
        'threshold_f1': float(best_threshold_f1),
        'threshold_fnr': float(best_threshold_fnr),
        'fnr': float(fn/(tp+fn+1e-8)),
        'fpr': float(fp/(fp+tn+1e-8)),
    }
    
    result_file = Path(args.save_dir) / f"results_{args.dataset}_{args.mvtec_category if args.dataset == 'mvtec' else 'wafer'}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] 结果已保存到: {result_file}")

    return auroc, best_f1, accuracy


def evaluate_all_mvtec(args):
    """评估MVTec AD所有类别"""
    categories = get_mvtec_categories()
    results = {}
    
    print(f"\n{'='*60}")
    print(f"评估MVTec AD所有类别 ({len(categories)}个)")
    print(f"{'='*60}\n")
    
    for category in categories:
        print(f"\n{'-'*40}")
        print(f"评估类别: {category}")
        print(f"{'-'*40}")
        
        args.mvtec_category = category
        args.checkpoint = str(Path(args.save_dir) / f"best_model_mvtec_{category}.pth")
        
        # 检查模型是否存在
        if not Path(args.checkpoint).exists():
            print(f"[WARNING] 模型不存在，跳过: {args.checkpoint}")
            continue
        
        try:
            auroc, f1, acc = evaluate(args)
            results[category] = {'auroc': auroc, 'f1': f1, 'accuracy': acc}
        except Exception as e:
            print(f"[ERROR] 评估失败: {e}")
            continue
    
    # 汇总结果
    if results:
        avg_auroc = np.mean([r['auroc'] for r in results.values()])
        avg_f1 = np.mean([r['f1'] for r in results.values()])
        
        print(f"\n{'='*60}")
        print(f"MVTec AD 汇总结果")
        print(f"{'='*60}")
        print(f"平均AUROC: {avg_auroc:.4f}")
        print(f"平均F1: {avg_f1:.4f}")
        print(f"\n各类别结果:")
        for cat, res in results.items():
            print(f"  {cat:15s}: AUROC={res['auroc']:.4f}, F1={res['f1']:.4f}")
        
        # 保存汇总结果
        summary_file = Path(args.save_dir) / "mvtec_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({'average': {'auroc': avg_auroc, 'f1': avg_f1}, 'per_category': results}, f, indent=2)
        print(f"\n[INFO] 汇总结果已保存到: {summary_file}")


# ============================================================
# 9. 主入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='ViT+MoCo改进版 - 支持MVTec AD')

    # 数据
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='晶圆数据集根目录')
    parser.add_argument('--mvtec_dir', type=str, default='./mvtec_anomaly_detection',
                       help='MVTec AD数据集路径')
    parser.add_argument('--dataset', type=str, default='wafer', choices=['wafer', 'mvtec'],
                       help='使用哪个数据集')
    parser.add_argument('--mvtec_category', type=str, default='bottle',
                       help='MVTec类别（当dataset=mvtec时）')
    parser.add_argument('--img_size', type=int, default=224,
                       help='输入图片大小')

    # 模型
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='ViT嵌入维度')
    parser.add_argument('--queue_size', type=int, default=1024,
                       help='MoCo队列大小')
    parser.add_argument('--momentum', type=float, default=0.999,
                       help='动量更新系数')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='对比损失温度')

    # 改进功能
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                       help='使用多尺度特征')
    parser.add_argument('--use_cutpaste', action='store_true', default=True,
                       help='使用CutPaste增强')
    parser.add_argument('--use_feature_generator', action='store_true', default=True,
                       help='使用特征生成器')
    parser.add_argument('--use_hypersphere', action='store_true', default=True,
                       help='使用超球面约束')

    # 损失权重
    parser.add_argument('--hypersphere_weight', type=float, default=0.1,
                       help='超球面损失权重')
    parser.add_argument('--discriminator_weight', type=float, default=0.05,
                       help='判别器损失权重')
    parser.add_argument('--generator_weight', type=float, default=0.05,
                       help='生成器损失权重')
    parser.add_argument('--cutpaste_prob', type=float, default=0.3,
                       help='CutPaste概率')

    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.03,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    # 保存/加载
    parser.add_argument('--save_dir', type=str, default='./checkpoints_v3',
                       help='模型保存目录')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='评估时加载的模型路径')

    # 评估
    parser.add_argument('--pca_components', type=int, default=None,
                       help='PCA维度')
    parser.add_argument('--eval_all', action='store_true',
                       help='评估MVTec所有类别')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例 (0.0-1.0)，如0.2表示从正常和缺陷样本各取20%%作为验证集')
    parser.add_argument('--score_mode', type=str, default='combined',
                       choices=['combined', 'mahal', 'memory', 'max'],
                       help='异常评分模式: combined=归一化融合, mahal=仅马氏距离, memory=仅记忆库, max=取最大')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'train_eval_all'],
                       help='运行模式')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("ViT + MoCo v2 改进版 V3 (模块化) - 支持MVTec AD验证")
    print("=" * 70)
    print("\n主体架构: ViT Encoder + MoCo v2 (保持不变)")
    print("增强功能:")
    print("  ✓ CutPaste 合成异常增强")
    print("  ✓ 超球面约束损失 (CFA)")
    print("  ✓ 特征生成-判别框架 (SimpleNet)")
    print("  ✓ 难负样本挖掘")
    print("  ✓ 记忆库辅助检测 (PatchCore风格)")
    print("=" * 70)
    
    # 打印数据路径信息
    print(f"\n[INFO] 当前工作目录: {Path.cwd()}")
    print(f"[INFO] 数据集: {args.dataset}")
    if args.dataset == 'wafer':
        data_path = Path(args.data_dir) / "数据集" / "数据集"
        print(f"[INFO] 晶圆数据路径: {data_path} (绝对路径: {data_path.absolute()})")
        print(f"[INFO] 路径是否存在: {data_path.exists()}")
    else:
        mvtec_path = Path(args.mvtec_dir)
        print(f"[INFO] MVTec路径: {mvtec_path} (绝对路径: {mvtec_path.absolute()})")
        print(f"[INFO] 路径是否存在: {mvtec_path.exists()}")
        print(f"[INFO] MVTec类别: {args.mvtec_category}")
    print("=" * 70)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        if not args.checkpoint:
            args.checkpoint = str(Path(args.save_dir) / f"best_model_{args.dataset}.pth")
        evaluate(args)
    elif args.mode == 'train_eval_all':
        # 训练并评估MVTec所有类别
        if args.dataset != 'mvtec':
            print("[ERROR] train_eval_all模式只支持mvtec数据集")
            return
        
        categories = get_mvtec_categories()
        for category in categories:
            print(f"\n{'='*70}")
            print(f"处理类别: {category}")
            print(f"{'='*70}")
            
            args.mvtec_category = category
            args.save_dir = f'./checkpoints_v3/mvtec_{category}'
            
            # 训练
            train(args)
            
            # 评估
            args.checkpoint = str(Path(args.save_dir) / f"best_model_mvtec.pth")
            evaluate(args)
        
        # 汇总所有结果
        args.eval_all = True
        evaluate_all_mvtec(args)


if __name__ == '__main__':
    main()
