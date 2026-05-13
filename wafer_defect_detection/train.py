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
运行方式：python -m wafer_defect_detection.train
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
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from wafer_defect_detection.utils import get_device, set_seed, SemiconductorTransform, EvalTransform
from wafer_defect_detection.data import (
    WaferDataset, WaferTrainDataset, WaferEvalDataset,
    MVTecDataset, MVTecTrainDataset, MVTecEvalDataset,
    get_mvtec_categories,
    PerCategoryWaferTrainDataset, PerCategoryWaferEvalDataset,
    get_wafer_categories,
)
from wafer_defect_detection.models import ViTEncoder, ImprovedMoCo
from wafer_defect_detection.detectors import AnomalyDetector


# ============================================================
# 7. 训练流程
# ============================================================
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def save_confusion_images(img_paths, gt_list, preds, save_root, class_name):
    """将测试图片按TP/FP/FN/TN分类保存到对应文件夹"""
    import shutil
    categories = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    for i, (img_path, gt, pred) in enumerate(zip(img_paths, gt_list, preds)):
        if gt == 1 and pred == 1:
            categories['TP'].append(img_path)
        elif gt == 0 and pred == 1:
            categories['FP'].append(img_path)
        elif gt == 1 and pred == 0:
            categories['FN'].append(img_path)
        elif gt == 0 and pred == 0:
            categories['TN'].append(img_path)
    for cat_name, paths in categories.items():
        cat_dir = os.path.join(save_root, class_name, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        for src_path in paths:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(cat_dir, fname)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"  [WARN] 复制失败 {src_path}: {e}")
    return categories


def train(args):
    """训练模型 - 支持验证集划分"""
    device = get_device()
    set_seed(args.seed)

    # 数据集
    if args.dataset == 'wafer':
        if args.wafer_category:
            # 按品类+视图训练
            data_root = Path(args.data_dir) / "晶圆分类数据集"
            transform = SemiconductorTransform(img_size=args.img_size,
                                               use_cutpaste=args.use_cutpaste,
                                               cutpaste_prob=args.cutpaste_prob)
            train_dataset = PerCategoryWaferTrainDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=transform,
            )
            val_dataset = PerCategoryWaferEvalDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=EvalTransform(img_size=args.img_size),
            )
            print(f"[INFO] 品类[{args.wafer_category}] 视图[{args.wafer_view}]")
            dataset = train_dataset
        else:
            data_root = Path(args.data_dir) / "晶圆分类数据集"
            transform = SemiconductorTransform(img_size=args.img_size, 
                                               use_cutpaste=args.use_cutpaste,
                                               cutpaste_prob=args.cutpaste_prob)
            
            if args.val_ratio > 0:
                train_dataset = WaferTrainDataset(
                    data_root, transform=transform,
                    val_ratio=args.val_ratio, split='train'
                )
                val_dataset = WaferEvalDataset(
                    data_root, transform=EvalTransform(img_size=args.img_size),
                    val_ratio=args.val_ratio, split='val'
                )
                print(f"[INFO] 使用验证集划分: val_ratio={args.val_ratio}")
                print(f"  - 训练集: {len(train_dataset)} 张")
                print(f"  - 验证集: {len(val_dataset)} 张")
                dataset = train_dataset
            else:
                dataset = WaferTrainDataset(data_root, transform=transform)
                val_dataset = None
            
    elif args.dataset == 'mvtec':
        data_root = Path(args.mvtec_dir)
        transform = SemiconductorTransform(img_size=args.img_size,
                                           use_cutpaste=args.use_cutpaste,
                                           cutpaste_prob=args.cutpaste_prob)
        dataset = MVTecTrainDataset(data_root, args.mvtec_category, transform=transform)
        val_dataset = MVTecEvalDataset(
            data_root, args.mvtec_category,
            transform=EvalTransform(img_size=args.img_size),
            phase='test'
        )
        print(f"[INFO] MVTec {args.mvtec_category} 训练: {len(dataset)} 张  测试: {len(val_dataset)} 张")
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type == 'cuda')
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

    # 加载预训练权重（如果指定）
    if args.pretrained:
        if Path(args.pretrained).exists():
            print(f"[INFO] 加载预训练权重: {args.pretrained}")
            model.encoder_q.load_pretrained(args.pretrained)
            # 同步到encoder_k
            for param_q, param_k in zip(model.encoder_q.parameters(),
                                       model.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
        else:
            print(f"[WARNING] 预训练权重不存在: {args.pretrained}")

    # 优化器 - 使用AdamW代替SGD（更稳定）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print(f"\n{'='*60}")
    print(f"开始训练 | 设备: {device} | 数据集: {args.dataset}")
    if args.dataset == 'wafer' and args.wafer_category:
        print(f"品类: {args.wafer_category} | 视图: {args.wafer_view}")
    elif args.dataset == 'wafer':
        print(f"品类: 全部品类混合训练（建议使用 --wafer_category 指定品类）")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"改进: CutPaste={args.use_cutpaste}, 特征生成器={args.use_feature_generator}, 超球面={args.use_hypersphere}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    best_auroc = 0.0
    save_dir = Path(args.save_dir)
    if args.dataset == 'wafer' and args.wafer_category:
        # 品类模型保存在子文件夹中
        cat_subdir = f"{args.wafer_category}_{args.wafer_view}".replace(' ', '_')
        save_dir = save_dir / cat_subdir
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 模型保存后缀
    if args.dataset == 'wafer' and args.wafer_category:
        model_tag = cat_subdir.lower()
    else:
        model_tag = args.dataset
    
    # TensorBoard
    writer = SummaryWriter(log_dir=save_dir / 'tensorboard' / model_tag)
    
    eval_interval = getattr(args, 'eval_interval', 25)
    print(f"[INFO] TensorBoard日志: {save_dir / 'tensorboard' / model_tag}")
    print(f"[INFO] 定期评估间隔: {eval_interval} epochs")
    
    # 用于定期评估的验证集dataloader
    _eval_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        _eval_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )

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
            # 梯度裁剪（防止训练不稳定）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += losses[k].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'temp': f"{model.temperature:.4f}"
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
        
        # TensorBoard日志
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/contrastive', loss_components['contrastive'], epoch)
        writer.add_scalar('Loss/hypersphere', loss_components['hypersphere'], epoch)
        writer.add_scalar('Loss/discriminator', loss_components['discriminator'], epoch)
        writer.add_scalar('Loss/generator', loss_components['generator'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # 保存最佳模型（基于loss）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_q_state_dict': model.encoder_q.state_dict(),
                'projector_q_state_dict': model.projector_q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, save_dir / f"best_model_{model_tag}.pth")
            print(f"  -> 保存最佳模型 (loss={best_loss:.4f})")
        
        # 定期评估
        if _eval_dataloader is not None and (epoch + 1) % eval_interval == 0:
            print(f"\n{'─'*40}")
            print(f"[EVAL] Epoch {epoch+1}/{args.epochs} 定期评估...")
            print(f"{'─'*40}")
            try:
                encoder_eval = model.encoder_q
                encoder_eval.eval()
                detector = AnomalyDetector(
                    device=device, n_components=args.pca_components,
                    use_hypersphere=args.use_hypersphere,
                    use_memory_bank=True, memory_ratio=0.1,
                    min_pca_components=32, pca_variance=0.995,
                    score_mode=getattr(args, 'score_mode', 'combined'),
                )
                # 用验证集拟合（val_dataset返回3值：img,label,path）
                # 注：fit()内部会过滤正常样本，所以即使val含异常样本也安全
                train_eval_loader = _eval_dataloader
                detector.fit(encoder_eval, train_eval_loader,
                             use_multiscale=args.use_multiscale)
                scores, labels, _ = detector.score(
                    encoder_eval, _eval_dataloader,
                    use_multiscale=args.use_multiscale)
                from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
                auroc = roc_auc_score(labels, scores)
                precision, recall, thresholds = precision_recall_curve(labels, scores)
                f1_s = 2 * precision * recall / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_s)
                best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
                best_f1 = f1_s[best_idx]
                preds_eval = (scores > best_thr).astype(int)
                acc = accuracy_score(labels, preds_eval)
                tp_e = ((preds_eval == 1) & (labels == 1)).sum()
                fp_e = ((preds_eval == 1) & (labels == 0)).sum()
                fn_e = ((preds_eval == 0) & (labels == 1)).sum()
                tn_e = ((preds_eval == 0) & (labels == 0)).sum()
                fnr_e = fn_e / (tp_e + fn_e + 1e-8)
                fpr_e = fp_e / (fp_e + tn_e + 1e-8)
                print(f'  AUROC:{auroc:.4f} | Acc:{acc:.4f} F1:{best_f1:.4f}')
                print(f'  混淆矩阵: [[TN={tn_e}, FP={fp_e}], [FN={fn_e}, TP={tp_e}]]')
                print(f'  漏检率(FNR):{fnr_e:.4f}  误检率(FPR):{fpr_e:.4f}')
                # TensorBoard
                writer.add_scalar('Metrics/AUROC', auroc, epoch)
                writer.add_scalar('Metrics/F1', best_f1, epoch)
                writer.add_scalar('Metrics/Accuracy', acc, epoch)
                writer.add_scalar('Metrics/FNR', fnr_e, epoch)
                writer.add_scalar('Metrics/FPR', fpr_e, epoch)
                # 保存AUROC最优模型
                if auroc > best_auroc:
                    best_auroc = auroc
                    torch.save({
                        'epoch': epoch,
                        'encoder_q_state_dict': model.encoder_q.state_dict(),
                        'projector_q_state_dict': model.projector_q.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'auroc': auroc,
                        'args': vars(args),
                    }, save_dir / f"best_auroc_{model_tag}.pth")
                    print(f"  >>> 保存AUROC最优模型 (AUROC={auroc:.4f})")
                model.train()
            except Exception as e:
                print(f"  [WARN] 定期评估失败: {e}")
                model.train()
            print(f"{'─'*40}\n")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'encoder_q_state_dict': model.encoder_q.state_dict(),
        'projector_q_state_dict': model.projector_q.state_dict(),
        'loss': avg_loss,
        'args': vars(args),
    }, save_dir / f"final_model_{model_tag}.pth")
    
    writer.close()
    
    print(f"\n训练完成! 最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")
    
    # ===== 最终评估 + 混淆矩阵图片保存 =====
    try:
        print(f"\n{'='*50}")
        print(f"[FINAL] 加载最佳模型进行最终评估...")
        print(f"{'='*50}")
        best_ckpt_path = save_dir / f"best_auroc_{model_tag}.pth"
        if not best_ckpt_path.exists():
            best_ckpt_path = save_dir / f"best_model_{model_tag}.pth"
        if best_ckpt_path.exists():
            b_encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
            b_ckpt = torch.load(str(best_ckpt_path), map_location='cpu', weights_only=False)
            b_encoder.load_state_dict(b_ckpt['encoder_q_state_dict'])
            b_encoder = b_encoder.to(device)
            b_encoder.eval()
            
            # 准备评估数据
            eval_transform = EvalTransform(img_size=args.img_size)
            if args.dataset == 'wafer' and args.wafer_category:
                data_root = Path(args.data_dir) / "晶圆分类数据集"
                train_eval = PerCategoryWaferEvalDataset(data_root, args.wafer_category,
                    view=args.wafer_view, transform=eval_transform)
                test_eval = PerCategoryWaferEvalDataset(data_root, args.wafer_category,
                    view=args.wafer_view, transform=eval_transform)
            elif args.dataset == 'mvtec':
                data_root = Path(args.mvtec_dir)
                train_eval = MVTecEvalDataset(data_root, args.mvtec_category,
                    transform=eval_transform, phase='train')
                test_eval = MVTecEvalDataset(data_root, args.mvtec_category,
                    transform=eval_transform, phase='test')
            else:
                if val_dataset is None:
                    print("[WARN] 未划分验证集，跳过最终评估")
                    return model
                train_eval = val_dataset
                test_eval = val_dataset
            
            train_el = DataLoader(train_eval, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
            test_el = DataLoader(test_eval, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
            
            b_detector = AnomalyDetector(
                device=device, n_components=args.pca_components,
                use_hypersphere=args.use_hypersphere,
                use_memory_bank=True, memory_ratio=0.1,
                min_pca_components=32, pca_variance=0.995,
                score_mode=getattr(args, 'score_mode', 'combined'),
            )
            b_detector.fit(b_encoder, train_el, use_multiscale=args.use_multiscale)
            scores, labels, paths = b_detector.score(b_encoder, test_el,
                use_multiscale=args.use_multiscale)
            
            from sklearn.metrics import roc_auc_score, accuracy_score
            auroc = roc_auc_score(labels, scores)
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            f1_s = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_s)
            best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            preds = (scores > best_thr).astype(int)
            cm_tp = ((preds == 1) & (labels == 1)).sum()
            cm_fp = ((preds == 1) & (labels == 0)).sum()
            cm_fn = ((preds == 0) & (labels == 1)).sum()
            cm_tn = ((preds == 0) & (labels == 0)).sum()
            cm_fnr = cm_fn / (cm_tp + cm_fn + 1e-8)
            cm_fpr = cm_fp / (cm_fp + cm_tn + 1e-8)
            
            print(f'\n最终评估结果:')
            print(f'  AUROC:{auroc:.4f} | Acc:{accuracy_score(labels, preds):.4f} F1:{f1_score(labels, preds):.4f}')
            print(f'  混淆矩阵: [[TN={cm_tn}, FP={cm_fp}], [FN={cm_fn}, TP={cm_tp}]]')
            print(f'  漏检率(FNR):{cm_fnr:.4f}  误检率(FPR):{cm_fpr:.4f}')
            
            # 保存混淆矩阵图片
            confusion_root = save_dir / 'confusion_images'
            class_name = model_tag
            cat_counts = save_confusion_images(paths, labels, preds,
                                                str(confusion_root), class_name)
            print(f"\n混淆矩阵图片已保存至: {confusion_root}/{class_name}/")
            for cat_name in ['TP', 'FP', 'FN', 'TN']:
                print(f"  {cat_name}: {len(cat_counts[cat_name])} 张")
            for cat_name in ['FP', 'FN']:
                if cat_counts[cat_name]:
                    print(f"  [{cat_name}] 共 {len(cat_counts[cat_name])} 张:")
                    for p in cat_counts[cat_name]:
                        print(f"    {os.path.basename(p)}")
        else:
            print(f"  [WARN] 未找到模型文件: {best_ckpt_path}")
    except Exception as e:
        print(f"  [WARN] 最终评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    return model


# ============================================================
# 8. 评估流程 - 支持MVTec AD
# ============================================================
def evaluate(args):
    """评估模型 - 支持验证集评估"""
    device = get_device()

    # 评估
    eval_transform = EvalTransform(img_size=args.img_size)
    if args.dataset == 'wafer':
        if args.wafer_category:
            data_root = Path(args.data_dir) / "晶圆分类数据集"
            train_dataset = PerCategoryWaferEvalDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=eval_transform,
            )
            eval_dataset = PerCategoryWaferEvalDataset(
                data_root, args.wafer_category, view=args.wafer_view,
                transform=eval_transform,
            )
        else:
            data_root = Path(args.data_dir) / "数据集" / "数据集"
            
            if args.val_ratio > 0:
                train_dataset = WaferEvalDataset(
                    data_root, transform=eval_transform, 
                    val_ratio=args.val_ratio, split='train'
                )
                eval_dataset = WaferEvalDataset(
                    data_root, transform=eval_transform,
                    val_ratio=args.val_ratio, split='val'
                )
                print(f"[INFO] 使用验证集评估: val_ratio={args.val_ratio}")
            else:
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
    if args.dataset == 'wafer' and args.wafer_category:
        cat_subdir = f"{args.wafer_category}_{args.wafer_view}".replace(' ', '_')
        model_tag = cat_subdir.lower()
        eval_save_dir = Path(args.save_dir) / cat_subdir
    else:
        model_tag = args.dataset
        eval_save_dir = Path(args.save_dir)
    
    # 自动检测checkpoint路径（如果未指定）
    if not args.checkpoint:
        args.checkpoint = str(eval_save_dir / f"best_model_{model_tag}.pth")
    
    encoder = ViTEncoder(img_size=args.img_size, embed_dim=args.embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_q_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    use_multiscale = checkpoint.get('args', {}).get('use_multiscale', True)
    print(f"[INFO] 加载模型: {args.checkpoint}")
    print(f"[INFO] 多尺度特征: {use_multiscale}")

    # 异常检测 - 使用训练集fit，测试集score
    detector = AnomalyDetector(
        device=device, 
        n_components=args.pca_components,
        use_hypersphere=args.use_hypersphere,
        use_memory_bank=True,
        memory_ratio=0.1,  # 增大到10%
        min_pca_components=32,
        pca_variance=0.995,
        score_mode=getattr(args, 'score_mode', 'combined'),
    )
    print(f"[INFO] 使用训练集拟合异常检测器...")
    detector.fit(encoder, train_loader, use_multiscale=use_multiscale)

    print(f"[INFO] 在测试集上评估...")
    scores, labels, paths = detector.score(encoder, eval_loader, use_multiscale=use_multiscale)

    # 计算指标
    auroc = roc_auc_score(labels, scores)
    
    # === 策略1: F1最优阈值 ===
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_thresh_idx]

    # === 策略2: FNR优先阈值 ===
    best_threshold_fnr = detector.find_threshold_fnr_priority(scores, labels, target_fnr=0.0)

    # 使用F1最优阈值作为主要结果（FNR优先往往导致FPR过高）
    best_threshold = best_threshold_f1

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
    print(f"FNR优先阈值: {best_threshold_fnr:.4f}")
    print(f"\n当前使用: F1最优阈值 = {best_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP={tp} FP={fp}")
    print(f"  FN={fn} TN={tn}")
    print(f"  漏检率(FNR): {fn/(tp+fn+1e-8):.4f}")
    print(f"  误检率(FPR): {fp/(fp+tn+1e-8):.4f}")
    print(f"{'='*60}")

    # 也算一下FNR优先阈值的准确率
    preds_fnr = (scores > best_threshold_fnr).astype(int)
    acc_fnr = accuracy_score(labels, preds_fnr)
    tp_fnr = ((preds_fnr == 1) & (labels == 1)).sum()
    fp_fnr = ((preds_fnr == 1) & (labels == 0)).sum()
    fn_fnr = ((preds_fnr == 0) & (labels == 1)).sum()
    tn_fnr = ((preds_fnr == 0) & (labels == 0)).sum()
    print(f"\n--- FNR优先阈值结果 ---")
    print(f"Accuracy: {acc_fnr:.4f}")
    print(f"  TP={tp_fnr} FP={fp_fnr}")
    print(f"  FN={fn_fnr} TN={tn_fnr}")
    print(f"  漏检率(FNR): {fn_fnr/(tp_fnr+fn_fnr+1e-8):.4f}")
    print(f"  误检率(FPR): {fp_fnr/(fp_fnr+tn_fnr+1e-8):.4f}")

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
    
    result_file = Path(args.save_dir) / f"results_{model_tag}.json"
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
        
        if not Path(args.checkpoint).exists():
            print(f"[WARNING] 模型不存在，跳过: {args.checkpoint}")
            continue
        
        try:
            auroc, f1, acc = evaluate(args)
            results[category] = {'auroc': auroc, 'f1': f1, 'accuracy': acc}
        except Exception as e:
            print(f"[ERROR] 评估失败: {e}")
            continue
    
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
    parser.add_argument('--wafer_category', type=str, default=None,
                       help='晶圆品类名称（当dataset=wafer且分品类训练时），如"BGA S5E 16x7"')
    parser.add_argument('--wafer_view', type=str, default='ALL',
                       choices=['ALL', 'UP', 'DOWN'],
                       help='晶圆视图：ALL/UP/DOWN（当dataset=wafer且分品类训练时）')
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
    parser.add_argument('--no-use_cutpaste', action='store_false', dest='use_cutpaste',
                       help='关闭CutPaste增强')
    parser.add_argument('--use_feature_generator', action='store_true', default=True,
                       help='使用特征生成器')
    parser.add_argument('--no-use_feature_generator', action='store_false', dest='use_feature_generator',
                       help='关闭特征生成器')
    parser.add_argument('--use_hypersphere', action='store_true', default=True,
                       help='使用超球面约束')
    parser.add_argument('--no-use_hypersphere', action='store_false', dest='use_hypersphere',
                       help='关闭超球面约束')

    # 损失权重
    parser.add_argument('--hypersphere_weight', type=float, default=0.1,
                       help='超球面损失权重')
    parser.add_argument('--discriminator_weight', type=float, default=0.05,
                       help='判别器损失权重')
    parser.add_argument('--generator_weight', type=float, default=0.05,
                       help='生成器损失权重')
    parser.add_argument('--cutpaste_prob', type=float, default=0.3,
                       help='CutPaste概率')

    # 训练 - 改进默认值
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率（AdamW推荐1e-3）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    # 预训练
    parser.add_argument('--pretrained', type=str, default=None,
                       help='预训练权重路径（可选）')

    # 保存/加载
    parser.add_argument('--save_dir', type=str, default='./checkpoints_v3_baseline/mycode',
                       help='模型保存目录（品类训练时自动创建子文件夹）')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='评估时加载的模型路径')

    # 评估
    parser.add_argument('--pca_components', type=int, default=None,
                       help='PCA维度')
    parser.add_argument('--score_mode', type=str, default='combined',
                       choices=['combined', 'mahal', 'memory', 'max'],
                       help='异常评分融合模式')
    parser.add_argument('--eval_all', action='store_true',
                       help='评估MVTec所有类别')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例 (0.0-1.0)，默认0.2')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='定期评估间隔epoch数（默认10，设为0关闭）')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'train_eval_all', 'train_all_wafer'],
                        help='运行模式: train | eval | train_eval_all(MVTec) | train_all_wafer(全部晶圆品类)')

    return parser.parse_args()


# ============================================================
# 全品类晶圆训练模式
# ============================================================
def train_all_wafer_modes(args):
    """遍历所有晶圆品类的UP/DOWN视图，逐个训练"""
    import copy

    data_root = Path(args.data_dir) / "晶圆分类数据集"
    categories = get_wafer_categories(str(data_root))
    views = ['UP', 'DOWN']

    print(f"\n{'='*70}")
    print(f"全品类晶圆训练: {len(categories)}个品类 × 2个视图 = {len(categories)*2}个模型")
    print(f"{'='*70}\n")

    results = {}
    for i, cat in enumerate(categories):
        for view in views:
            cat_view = f"{cat}_{view}"
            cat_subdir = cat_view.replace(' ', '_')
            model_path = Path(args.save_dir) / cat_subdir
            print(f"\n{'─'*60}")
            print(f"[{i+1}/{len(categories)}] 🏭 {cat} | 视图: {view}")
            print(f"    模型保存: {model_path}/")
            print(f"{'─'*60}")

            cat_args = copy.deepcopy(args)
            cat_args.dataset = 'wafer'
            cat_args.wafer_category = cat
            cat_args.wafer_view = view
            cat_args.mode = 'train'

            try:
                model = train(cat_args)
                results[cat_view] = 'trained'
                print(f"  ✅ {cat_view} 训练完成")
            except Exception as e:
                print(f"  ❌ {cat_view} 训练失败: {e}")
                results[cat_view] = f'failed: {e}'

    # 汇总
    success = sum(1 for v in results.values() if v == 'trained')
    failed = sum(1 for v in results.values() if 'failed' in str(v))
    print(f"\n{'='*70}")
    print(f"全品类训练完成! 成功: {success}, 失败: {failed}")
    print(f"{'='*70}")

    # 保存汇总
    summary_file = Path(args.save_dir) / "wafer_all_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 汇总保存: {summary_file}")


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
        data_path = Path(args.data_dir) / "晶圆分类数据集"
        print(f"[INFO] 晶圆数据路径: {data_path} (绝对路径: {data_path.absolute()})")
        print(f"[INFO] 路径是否存在: {data_path.exists()}")
        if data_path.exists():
            # 快速统计样本数
            from wafer_defect_detection.data.wafer_dataset import _collect_wafer_samples
            normal, defect = _collect_wafer_samples(data_path)
            print(f"[INFO] 正常样本: {len(normal)}, 缺陷样本: {len(defect)}")
    else:
        mvtec_path = Path(args.mvtec_dir)
        print(f"[INFO] MVTec路径: {mvtec_path} (绝对路径: {mvtec_path.absolute()})")
        print(f"[INFO] 路径是否存在: {mvtec_path.exists()}")
        print(f"[INFO] MVTec类别: {args.mvtec_category}")
    print("=" * 70)

    if args.mode == 'train':
        if args.dataset == 'wafer' and not args.wafer_category:
            # 未指定品类 → 遍历所有品类逐个训练（每个品类含UP/DOWN）
            train_all_wafer_modes(args)
        else:
            train(args)
    elif args.mode == 'eval':
        if not args.checkpoint:
            if args.dataset == 'wafer' and args.wafer_category:
                cat_subdir = f"{args.wafer_category}_{args.wafer_view}".replace(' ', '_')
                model_tag = cat_subdir.lower()
                args.checkpoint = str(Path(args.save_dir) / cat_subdir / f"best_model_{model_tag}.pth")
            else:
                model_tag = args.dataset
                args.checkpoint = str(Path(args.save_dir) / f"best_model_{model_tag}.pth")
        evaluate(args)
    elif args.mode == 'train_all_wafer':
        if args.dataset != 'wafer':
            print("[ERROR] train_all_wafer模式只支持wafer数据集")
            return
        train_all_wafer_modes(args)
    elif args.mode == 'train_eval_all':
        if args.dataset != 'mvtec':
            print("[ERROR] train_eval_all模式只支持mvtec数据集")
            return
        
        categories = get_mvtec_categories()
        for category in categories:
            print(f"\n{'='*70}")
            print(f"处理类别: {category}")
            print(f"{'='*70}")
            
            args.mvtec_category = category
            args.save_dir = f'./checkpoints_v3_baseline/mycode/mvtec_{category}'
            
            train(args)
            
            args.checkpoint = str(Path(args.save_dir) / f"best_model_mvtec.pth")
            evaluate(args)
        
        args.eval_all = True
        evaluate_all_mvtec(args)


if __name__ == '__main__':
    main()
