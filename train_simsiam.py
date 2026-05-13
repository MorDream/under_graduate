"""
DenseSimSiam训练+评估脚本
=========================

对比学习方案: DenseSimSiam (SimSiam + 稠密多尺度对比)
数据集: 晶圆半导体缺陷检测

运行方式:
  训练: python train_simsiam.py --mode train
  评估: python train_simsiam.py --mode eval
  一条龙: python train_simsiam.py --mode all

核心设计:
  1. SimSiam: 无动量编码器、无负样本队列、stop-gradient防坍缩
  2. Dense Contrast: patch级特征对比，学习局部正常模式
  3. Multi-Scale: 多层ViT特征SimSiam对齐
  4. 辅助: 超球面约束 + 特征生成-判别
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score
from tqdm import tqdm

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from wafer_defect_detection.utils import get_device, set_seed, SemiconductorTransform, EvalTransform
from wafer_defect_detection.data import (
    WaferTrainDataset, WaferEvalDataset,
    MVTecTrainDataset, MVTecEvalDataset, get_mvtec_categories,
    PerCategoryWaferTrainDataset, PerCategoryWaferEvalDataset,
    get_wafer_categories,
)
from wafer_defect_detection.models import ViTEncoder, DenseSimSiam
from wafer_defect_detection.detectors import AnomalyDetector


def parse_args():
    parser = argparse.ArgumentParser(description='DenseSimSiam 对比学习 - 晶圆缺陷检测')

    # --- 数据 ---
    parser.add_argument('--dataset', default='wafer', choices=['wafer', 'mvtec'],
                        help='数据集: wafer | mvtec')
    parser.add_argument('--data_dir', default='./data', help='晶圆数据集根目录')
    parser.add_argument('--mvtec_dir', default='./mvtec_anomaly_detection',
                        help='MVTec AD数据集路径')
    parser.add_argument('--mvtec_category', default='bottle',
                        help='MVTec类别 (--dataset mvtec时使用)')
    parser.add_argument('--wafer_category', default=None,
                        help='晶圆品类名称（当dataset=wafer且分品类训练时），如"BGA S5E 16x7"')
    parser.add_argument('--wafer_view', default='ALL', choices=['ALL', 'UP', 'DOWN'],
                        help='晶圆视图：ALL/UP/DOWN（当dataset=wafer且分品类训练时）')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例(正常和缺陷各取20%%)')

    # --- 模型 ---
    parser.add_argument('--embed_dim', type=int, default=384, help='ViT嵌入维度')
    parser.add_argument('--proj_dim', type=int, default=256, help='SimSiam投影维度')
    parser.add_argument('--pred_hidden', type=int, default=128, help='SimSiam预测器隐藏维度')
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                        help='多尺度SimSiam')
    parser.add_argument('--use_dense', action='store_true', default=True,
                        help='稠密(patch级)SimSiam')
    parser.add_argument('--use_feature_generator', action='store_true', default=True,
                        help='特征生成-判别框架')
    parser.add_argument('--use_hypersphere', action='store_true', default=True,
                        help='超球面约束')

    # --- 损失权重 ---
    parser.add_argument('--global_weight', type=float, default=1.0, help='全局SimSiam权重')
    parser.add_argument('--dense_weight', type=float, default=0.3, help='稠密SimSiam权重')
    parser.add_argument('--multiscale_weight', type=float, default=0.3, help='多尺度SimSiam权重')
    parser.add_argument('--hypersphere_weight', type=float, default=0.1)
    parser.add_argument('--discriminator_weight', type=float, default=0.05)
    parser.add_argument('--generator_weight', type=float, default=0.05)

    # --- 增强 ---
    parser.add_argument('--use_cutpaste', action='store_true', default=True,
                        help='使用CutPaste合成异常增强')
    parser.add_argument('--cutpaste_prob', type=float, default=0.3)

    # --- 训练 ---
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='AdamW学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # --- 保存/加载 ---
    parser.add_argument('--save_dir', default='./checkpoints_simsiam', help='模型保存目录')
    parser.add_argument('--checkpoint', default='', help='评估时加载的模型路径')
    parser.add_argument('--pretrained', default=None, help='预训练encoder权重路径')

    # --- 评估 ---
    parser.add_argument('--pca_components', type=int, default=None, help='PCA维度(默认自动)')
    parser.add_argument('--score_mode', default='combined',
                        choices=['combined', 'mahal', 'memory', 'max'],
                        help='异常评分融合模式')

    # --- 模式 ---
    parser.add_argument('--mode', default='train',
                        choices=['train', 'eval', 'all', 'train_eval_all', 'train_all_wafer'],
                        help='运行模式: train | eval | all | train_eval_all(MVTec) | train_all_wafer(全部晶圆)')

    return parser.parse_args()


def get_dataloaders(args, device):
    """准备训练集和验证集的数据加载器"""
    train_transform = SemiconductorTransform(
        img_size=args.img_size,
        use_cutpaste=args.use_cutpaste,
        cutpaste_prob=args.cutpaste_prob,
    )
    eval_transform = EvalTransform(img_size=args.img_size)

    if args.dataset == 'wafer':
        if args.wafer_category:
            data_root = Path(args.data_dir) / "晶圆分类数据集"
            train_dataset = PerCategoryWaferTrainDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=train_transform,
            )
            val_dataset = PerCategoryWaferEvalDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=eval_transform,
            )
            fit_dataset = PerCategoryWaferEvalDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=eval_transform,
            )
        else:
            data_root = Path(args.data_dir) / "数据集" / "数据集"
            if args.val_ratio > 0:
                train_dataset = WaferTrainDataset(
                    data_root, transform=train_transform,
                    val_ratio=args.val_ratio, split='train'
                )
                val_dataset = WaferEvalDataset(
                    data_root, transform=eval_transform,
                    val_ratio=args.val_ratio, split='val'
                )
                fit_dataset = WaferEvalDataset(
                    data_root, transform=eval_transform,
                    val_ratio=args.val_ratio, split='train'
                )
            else:
                train_dataset = WaferTrainDataset(data_root, transform=train_transform)
                val_dataset = WaferEvalDataset(data_root, transform=eval_transform)
                fit_dataset = WaferEvalDataset(data_root, transform=eval_transform)

    elif args.dataset == 'mvtec':
        data_root = Path(args.mvtec_dir)
        train_dataset = MVTecTrainDataset(data_root, args.mvtec_category,
                                          transform=train_transform)
        val_dataset = MVTecEvalDataset(data_root, args.mvtec_category,
                                       transform=eval_transform, phase='test')
        fit_dataset = MVTecEvalDataset(data_root, args.mvtec_category,
                                       transform=eval_transform, phase='train')
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    fit_loader = DataLoader(
        fit_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, fit_loader, val_loader


def train(args):
    """DenseSimSiam 训练流程"""
    device = get_device()
    set_seed(args.seed)

    # --- 数据 ---
    train_loader, fit_loader, val_loader = get_dataloaders(args, device)
    print(f"[INFO] 训练样本: {len(train_loader.dataset)} 张 (仅正常)")
    print(f"[INFO] 验证样本: {len(val_loader.dataset)} 张 (正常+缺陷)")

    # --- 模型 ---
    model = DenseSimSiam(
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        proj_dim=args.proj_dim,
        pred_hidden=args.pred_hidden,
        use_multiscale=args.use_multiscale,
        use_dense=args.use_dense,
        use_feature_generator=args.use_feature_generator,
        use_hypersphere=args.use_hypersphere,
    ).to(device)

    # 加载预训练权重
    if args.pretrained and Path(args.pretrained).exists():
        print(f"[INFO] 加载预训练encoder: {args.pretrained}")
        model.load_pretrained_encoder(args.pretrained)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[INFO] 总参数量: {total_params:.2f}M, 可训练: {trainable_params:.2f}M")

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # --- 训练循环 ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"DenseSimSiam 训练开始")
    ds_info = f"设备: {device} | 数据集: {args.dataset}"
    if args.dataset == 'mvtec':
        ds_info += f" ({args.mvtec_category})"
    elif args.wafer_category:
        ds_info += f" ({args.wafer_category}_{args.wafer_view})"
    print(ds_info)
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"模块: Multiscale={args.use_multiscale} Dense={args.use_dense}")
    print(f"辅助: Hypersphere={args.use_hypersphere} FeatGen={args.use_feature_generator}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    suffix = f"_{args.dataset}"
    if args.dataset == 'mvtec':
        suffix += f"_{args.mvtec_category}"
    elif args.wafer_category:
        suffix += f"_{args.wafer_category}_{args.wafer_view}"

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        loss_track = {'global': 0, 'dense': 0, 'multiscale': 0,
                      'hypersphere': 0, 'discriminator': 0, 'generator': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img_q, img_k in pbar:
            img_q = img_q.to(device)
            img_k = img_k.to(device)

            losses, _ = model(img_q, img_k, epoch=epoch, total_epochs=args.epochs)

            # 组合损失
            total_loss = (
                args.global_weight * losses['global'] +
                args.dense_weight * losses['dense'] +
                args.multiscale_weight * losses['multiscale'] +
                args.hypersphere_weight * losses['hypersphere'] +
                args.discriminator_weight * losses['discriminator'] +
                args.generator_weight * losses['generator']
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            for k in loss_track:
                loss_track[k] += losses[k].item()

            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        scheduler.step()
        n_batches = len(train_loader)
        epoch_loss /= n_batches
        for k in loss_track:
            loss_track[k] /= n_batches

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {epoch_loss:.4f} | "
              f"G:{loss_track['global']:.3f} D:{loss_track['dense']:.3f} "
              f"MS:{loss_track['multiscale']:.3f} | "
              f"H:{loss_track['hypersphere']:.3f} "
              f"Dsc:{loss_track['discriminator']:.3f} "
              f"Gen:{loss_track['generator']:.3f} | "
              f"LR: {lr:.6f}")

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = save_dir / f"best_model{suffix}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  → 最佳模型已保存: {checkpoint_path}")

    # 保存最终模型
    final_path = save_dir / f"final_model{suffix}.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'loss': epoch_loss,
        'args': vars(args),
    }, final_path)
    print(f"\n训练完成! 最佳Loss: {best_loss:.4f}")
    print(f"最终模型: {final_path}")

    return model


def evaluate(args):
    """DenseSimSiam 评估流程"""
    device = get_device()

    # --- 数据 ---
    _, fit_loader, val_loader = get_dataloaders(args, device)

    # --- 加载模型 ---
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        suffix = f"_{args.dataset}"
        if args.dataset == 'mvtec':
            suffix += f"_{args.mvtec_category}"
        elif args.wafer_category:
            suffix += f"_{args.wafer_category}_{args.wafer_view}"
        checkpoint_path = str(Path(args.save_dir) / f"best_model{suffix}.pth")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"[INFO] 加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 加载完整DenseSimSiam模型
    ckpt_args = ckpt.get('args', {})
    model = DenseSimSiam(
        embed_dim=ckpt_args.get('embed_dim', args.embed_dim),
        img_size=ckpt_args.get('img_size', args.img_size),
        proj_dim=ckpt_args.get('proj_dim', args.proj_dim),
        pred_hidden=ckpt_args.get('pred_hidden', args.pred_hidden),
        use_multiscale=ckpt_args.get('use_multiscale', args.use_multiscale),
        use_dense=ckpt_args.get('use_dense', args.use_dense),
        use_feature_generator=False,  # 评估不需要
        use_hypersphere=False,   # 评估不需要
    ).to(device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        # 兼容旧格式: 只保存了encoder
        model.encoder.load_state_dict(ckpt['encoder_q_state_dict'], strict=False)

    model.eval()

    use_multiscale = model.use_multiscale
    print(f"[INFO] 多尺度: {use_multiscale}, 稠密: {model.use_dense}")

    # --- 异常检测器 ---
    detector = AnomalyDetector(
        device=device,
        n_components=args.pca_components,
        use_hypersphere=True,
        use_memory_bank=True,
        memory_ratio=0.1,
        min_pca_components=32,
        pca_variance=0.995,
        score_mode=args.score_mode,
    )

    print("[INFO] 拟合异常检测器...")
    detector.fit(model, fit_loader, use_multiscale=use_multiscale)

    print("[INFO] 在验证集上评估...")
    scores, labels, _ = detector.score(model, val_loader, use_multiscale=use_multiscale)

    # --- 计算指标 ---
    auroc = roc_auc_score(labels, scores)

    # F1最优阈值
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    # FNR优先阈值
    fnr_thresh = detector.find_threshold_fnr_priority(scores, labels, target_fnr=0.0)

    # 混淆矩阵 (用F1最优阈值)
    preds = (scores > best_thresh).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    accuracy = accuracy_score(labels, preds)
    fnr = fn / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    # 打印结果
    print(f"\n{'='*60}")
    ds_label = f"Wafer" if args.dataset == 'wafer' else f"MVTec-AD ({args.mvtec_category})"
    print(f"DenseSimSiam 评估结果 - {ds_label}")
    print(f"{'='*60}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {best_f1:.4f} @ threshold={best_thresh:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP={tp:3d}  FP={fp:3d}")
    print(f"  FN={fn:3d}  TN={tn:3d}")
    print(f"  漏检率(FNR): {fnr:.4f}  |  误检率(FPR): {fpr:.4f}")

    # FNR优先阈值结果
    preds_fnr = (scores > fnr_thresh).astype(int)
    tp_f = ((preds_fnr == 1) & (labels == 1)).sum()
    fp_f = ((preds_fnr == 1) & (labels == 0)).sum()
    fn_f = ((preds_fnr == 0) & (labels == 1)).sum()
    tn_f = ((preds_fnr == 0) & (labels == 0)).sum()
    print(f"\nFNR优先阈值={fnr_thresh:.4f}:")
    print(f"  TP={tp_f:3d} FP={fp_f:3d} FN={fn_f:3d} TN={tn_f:3d}")
    print(f"  FNR={fn_f/(tp_f+fn_f+1e-8):.4f} FPR={fp_f/(fp_f+tn_f+1e-8):.4f}")
    print(f"{'='*60}")

    # 保存结果
    results = {
        'method': 'DenseSimSiam',
        'dataset': args.dataset,
        'category': args.mvtec_category if args.dataset == 'mvtec' else None,
        'auroc': float(auroc),
        'f1': float(best_f1),
        'accuracy': float(accuracy),
        'threshold_f1': float(best_thresh),
        'threshold_fnr': float(fnr_thresh),
        'fnr': float(fnr),
        'fpr': float(fpr),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    result_name = f"results_{args.dataset}"
    if args.dataset == 'mvtec':
        result_name += f"_{args.mvtec_category}"
    elif args.wafer_category:
        result_name += f"_{args.wafer_category}_{args.wafer_view}"
    result_file = Path(args.save_dir) / f"{result_name}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已保存: {result_file}")

    return auroc, best_f1


def evaluate_all_mvtec(args):
    """训练并评估MVTec AD所有15个类别"""
    import copy
    categories = get_mvtec_categories()
    results = {}

    print(f"\n{'='*60}")
    print(f"DenseSimSiam 全类别评估 - MVTec AD ({len(categories)}个类别)")
    print(f"{'='*60}\n")

    for i, category in enumerate(categories):
        print(f"\n{'─'*50}")
        print(f"[{i+1}/{len(categories)}] 处理: {category}")
        print(f"{'─'*50}")

        cat_args = copy.deepcopy(args)
        cat_args.dataset = 'mvtec'
        cat_args.mvtec_category = category

        # 训练
        model = train(cat_args)

        # 评估
        try:
            auroc, f1 = evaluate(cat_args)
            results[category] = {'auroc': float(auroc), 'f1': float(f1)}
            print(f"  → {category}: AUROC={auroc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"  [ERROR] {category} 评估失败: {e}")
            results[category] = {'auroc': 0, 'f1': 0, 'error': str(e)}

    # 汇总
    valid = {k: v for k, v in results.items() if v.get('auroc', 0) > 0}
    if valid:
        avg_auroc = np.mean([v['auroc'] for v in valid.values()])
        avg_f1 = np.mean([v['f1'] for v in valid.values()])

        print(f"\n{'='*60}")
        print(f"MVTec AD 汇总结果 (DenseSimSiam)")
        print(f"{'='*60}")
        print(f"平均AUROC: {avg_auroc:.4f}  |  平均F1: {avg_f1:.4f}")
        print(f"\n各类别:")
        for cat, res in results.items():
            status = f"AUROC={res.get('auroc', 0):.4f}" if res.get('auroc', 0) > 0 else "FAILED"
            print(f"  {cat:15s}: {status}")
        print(f"{'='*60}")

        summary_file = Path(args.save_dir) / "mvtec_summary_simsiam.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': 'DenseSimSiam',
                'average': {'auroc': float(avg_auroc), 'f1': float(avg_f1)},
                'per_category': results
            }, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 汇总保存: {summary_file}")


def train_all_wafer_modes(args):
    """遍历所有晶圆品类的UP/DOWN视图，逐个训练 (DenseSimSiam)"""
    import copy

    data_root = Path(args.data_dir) / "晶圆分类数据集"
    categories = get_wafer_categories(str(data_root))
    views = ['UP', 'DOWN']

    print(f"\n{'='*70}")
    print(f"全品类晶圆训练 (DenseSimSiam): {len(categories)}个品类 × {len(views)}个视图")
    print(f"{'='*70}\n")

    results = {}
    for i, cat in enumerate(categories):
        for view in views:
            cat_view = f"{cat}_{view}"
            print(f"\n{'─'*60}")
            print(f"[{i+1}/{len(categories)}] 训练: {cat_view}")
            print(f"{'─'*60}")

            cat_args = copy.deepcopy(args)
            cat_args.dataset = 'wafer'
            cat_args.wafer_category = cat
            cat_args.wafer_view = view

            try:
                train(cat_args)
                # 自动评估
                try:
                    cat_args.checkpoint = ''
                    evaluate(cat_args)
                except Exception as e:
                    print(f"  ⚠️ {cat_view} 评估失败: {e}")
                results[cat_view] = 'done'
                print(f"  ✅ {cat_view} 完成")
            except Exception as e:
                print(f"  ❌ {cat_view} 训练失败: {e}")
                results[cat_view] = f'failed: {e}'

    success = sum(1 for v in results.values() if v == 'done')
    failed = sum(1 for v in results.values() if 'failed' in str(v))
    print(f"\n{'='*70}")
    print(f"全品类训练完成! 成功: {success}, 失败: {failed}")
    print(f"{'='*70}")

    summary_file = Path(args.save_dir) / "wafer_all_results_simsiam.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 汇总保存: {summary_file}")


def main():
    args = parse_args()

    print("=" * 70)
    print("DenseSimSiam: 稠密多尺度SimSiam对比学习")
    print("=" * 70)
    print("核心方法: SimSiam (无动量编码器, 无负样本队列, stop-gradient)")
    print("创新点:")
    print("  ✓ 稠密(patch级)对比: 学习局部正常模式")
    print("  ✓ 多尺度SimSiam: 多层ViT特征对齐")
    print("  ✓ 全局SimSiam: [CLS] token特征对比")
    print("  ✓ 超球面约束 + 特征生成-判别")
    print("对比MoCo:")
    print("  - 无momentum encoder → 节省50%显存")
    print("  - 无负样本队列 → 训练更简单稳定")
    print("  - stop-gradient → 自然防止坍缩")
    print("=" * 70)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'all':
        train(args)
        evaluate(args)
    elif args.mode == 'train_eval_all':
        if args.dataset != 'mvtec':
            print("[ERROR] train_eval_all 只支持 --dataset mvtec")
            return
        evaluate_all_mvtec(args)
    elif args.mode == 'train_all_wafer':
        if args.dataset != 'wafer':
            print("[ERROR] train_all_wafer 只支持 --dataset wafer")
            return
        train_all_wafer_modes(args)


if __name__ == '__main__':
    main()
