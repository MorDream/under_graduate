"""
ReContrast with ViT + DINOv2 Pretrained Weights
支持 MVTec AD 和晶圆数据集，使用ViT编码器 + DINOv2预训练权重

用法:
  # MVTec
  python recontrast_vit_wafer.py --dataset mvtec --categories bottle,capsule

  # 晶圆单个品类
  python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

  # 晶圆多个品类
  python recontrast_vit_wafer.py --dataset wafer --categories "BGA S5E 16x7,ESSD 12x5"
  
  # 使用现有ViT编码器 (不用DINOv2)
  python recontrast_vit_wafer.py --dataset wafer --use_wafer_encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import glob
import argparse
import copy
import logging
import warnings
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# 导入数据处理
from recontrast.dataset import get_data_transforms, get_strong_transforms, MVTecDataset, cut_paste
from recontrast.utils import (
    setup_seed, get_device, evaluation, visualize,
    global_cosine, global_cosine_hm, cal_anomaly_map,
    return_best_thr, min_max_norm, cvt2heatmap, show_cam_on_image,
    compute_pro
)

# 导入ViT版本ReContrast
from recontrast.models.recontrast_vit import (
    ReContrastViT, BottleneckViT, DecoderViT,
    load_dinov2_encoder, build_recontrast_vit
)

# 导入现有ViTEncoder (可选)
try:
    from wafer_defect_detection.models.vit_encoder import ViTEncoder
    HAS_WAFER_ENCODER = True
except ImportError:
    HAS_WAFER_ENCODER = False


from functools import partial


def modify_grad(x, inds, factor=0.):
    """抑制易重建token的梯度，迫使模型关注难例"""
    mask_float = inds.float()
    scale = 1.0 - mask_float + mask_float * factor
    return x * scale.unsqueeze(-1)  # [B,N] → [B,N,1] 对齐 [B,N,C]


def global_cosine_hm_tokens(a_list, b_list, alpha=1.0, factor=0.):
    """
    ViT版的交叉重建损失+难例挖掘（适配[B, N, C] token特征）
    原版global_cosine_hm的token适配版
    """
    cos_loss = torch.nn.CosineSimilarity(dim=-1)
    loss = 0
    for item in range(len(a_list)):
        a_ = a_list[item].detach()  # 固定target
        b_ = b_list[item]
        
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_)  # [B, N] per-token距离
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()
        
        loss += torch.mean(1 - cos_loss(a_, b_))
        
        # 难例挖掘: 只对距离超过阈值的token保留梯度
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    
    return loss


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


# ============================================================
# 晶圆数据集
# ============================================================
class WaferDataset(torch.utils.data.Dataset):
    """晶圆评估数据集 — 加载 test/good 和 test/defect"""
    def __init__(self, root, category, view, transform, gt_transform):
        self.root = Path(root) / category
        self.view = view
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        img_paths, labels = [], []
        test_dir = self.root / 'test'
        for subdir in ['good', 'defect']:
            label = 0 if subdir == 'good' else 1
            dir_path = test_dir / subdir
            if not dir_path.exists():
                continue
            for f in sorted(dir_path.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    if self.view and self.view != 'ALL':
                        if f'_{self.view}' not in f.stem and f'_{self.view}' not in f.name:
                            continue
                    img_paths.append(str(f))
                    labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # 晶圆没有像素级GT，用全零tensor代替
        gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        return img, gt, label, img_path


class WaferTrainDataset(torch.utils.data.Dataset):
    """晶圆训练数据集 — 加载 train/good"""
    def __init__(self, root, category, view, transform):
        self.root = Path(root) / category
        self.view = view
        self.transform = transform
        self.img_paths = self._load_dataset()

    def _load_dataset(self):
        img_paths = []
        train_dir = self.root / 'train' / 'good'
        if not train_dir.exists():
            return img_paths
        for f in sorted(train_dir.iterdir()):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                if self.view and self.view != 'ALL':
                    if f'_{self.view}' not in f.stem and f'_{self.view}' not in f.name:
                        continue
                img_paths.append(str(f))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0


# ============================================================
# ViT特征处理工具
# ============================================================
def extract_vit_features(encoder_output):
    """
    从ViT输出提取特征列表
    适配不同类型的encoder输出
    """
    if isinstance(encoder_output, tuple):
        # (final_feat, intermediate_features)
        final_feat, intermediates = encoder_output
        features = [final_feat] + intermediates
    elif isinstance(encoder_output, list):
        features = encoder_output
    else:
        # 单个输出
        features = [encoder_output]
    return features


def vit_features_to_anomaly_map(en_features, de_features, img_size=256):
    """
    将ViT特征转换为异常分数图
    
    Args:
        en_features: encoder输出特征列表
        de_features: decoder重建特征列表
        img_size: 输出图像尺寸
    
    Returns:
        anomaly_map: [H, W] 异常分数图
        anomaly_score: 标量异常分数
    """
    # 计算每层特征的重构误差
    total_error = 0
    weights = []
    
    for i, (en, de) in enumerate(zip(en_features, de_features)):
        # en/de: [B, N, C] 或 [B, C, H, W]
        if len(en.shape) == 3:
            # ViT格式: [B, N, C]
            # 计算每个token的误差
            error = F.mse_loss(en, de, reduction='none').mean(dim=-1)  # [B, N]
            # 平均所有tokens
            layer_error = error.mean()
        else:
            # CNN格式: [B, C, H, W]
            error = F.mse_loss(en, de, reduction='none').mean(dim=1)  # [B, H, W]
            layer_error = error.mean()
        
        # 深层权重更高
        weight = 0.1 + 0.9 * (i / max(1, len(en_features) - 1))
        weights.append(weight)
        total_error += weight * layer_error
    
    # 归一化权重
    total_weight = sum(weights)
    anomaly_score = (total_error / total_weight).item()
    
    # 生成简单的异常图 (基于最后一层)
    last_en = en_features[-1]
    last_de = de_features[-1]
    
    if len(last_en.shape) == 3:
        # ViT格式: [B, N, C]
        B, N, C = last_en.shape
        # 判断是否包含CLS/prefix token（patch tokens数应为平方数）
        H_patches = int(np.sqrt(N))
        if H_patches ** 2 == N:
            # 不包含CLS，直接用所有patch tokens
            en_patches = last_en
            de_patches = last_de
        else:
            # 包含CLS token，去掉第一个
            en_patches = last_en[:, 1:, :]
            de_patches = last_de[:, 1:, :]
            N = N - 1
            H_patches = int(np.sqrt(N))
        
        H = W = H_patches
        
        # 计算patch级余弦距离（与原版cal_anomaly_map一致）
        en_norm = F.normalize(en_patches, dim=-1)
        de_norm = F.normalize(de_patches, dim=-1)
        cos_sim = (en_norm * de_norm).sum(dim=-1)  # [B, N]
        error_patches = 1 - cos_sim  # [B, N]
        error_map = error_patches.reshape(B, 1, H, W)
        
        # 上采样到原图大小
        anomaly_map = F.interpolate(error_map, size=(img_size, img_size), 
                                    mode='bilinear', align_corners=False)
        anomaly_map = anomaly_map[0, 0].detach().cpu().numpy()
    else:
        # CNN格式
        error_map = F.mse_loss(last_en, last_de, reduction='none').mean(dim=1, keepdim=True)
        anomaly_map = F.interpolate(error_map, size=(img_size, img_size),
                                    mode='bilinear', align_corners=False)
        anomaly_map = anomaly_map[0, 0].detach().cpu().numpy()
    
    return anomaly_map, anomaly_score


# ============================================================
# 评估函数
# ============================================================
def evaluate_full(model, dataloader, device, _class_=None, return_details=False):
    """完整评估：AUROC + 混淆矩阵 + 准确率 + F1"""
    model.eval()
    gt_list_sp = []
    pr_list_sp = []
    img_paths = []

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)
            
            # 使用ViT特定的异常图计算
            anomaly_map, sp_score = vit_features_to_anomaly_map(en, de, img_size=img.shape[-1])
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            sp_score = anomaly_map.max()
            
            gt_list_sp.append(label.item())
            pr_list_sp.append(sp_score)
            img_paths.append(img_path[0] if isinstance(img_path, (list, tuple)) else img_path)

    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    best_thr = return_best_thr(gt_list_sp, pr_list_sp)
    preds = (np.array(pr_list_sp) >= best_thr).astype(int)
    cm = confusion_matrix(gt_list_sp, preds)
    acc = accuracy_score(gt_list_sp, preds)
    f1 = f1_score(gt_list_sp, preds)

    if return_details:
        return 0, auroc_sp, 0, cm, acc, f1, best_thr, img_paths, gt_list_sp, pr_list_sp, preds
    return 0, auroc_sp, 0, cm, acc, f1, best_thr


def save_confusion_images(img_paths, gt_list, preds, save_root, _class_):
    """将测试图片按 TP/FP/FN/TN 分类保存到对应文件夹"""
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
        cat_dir = os.path.join(save_root, _class_, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        for src_path in paths:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(cat_dir, fname)
            shutil.copy2(src_path, dst_path)

    return categories


# ============================================================
# 训练函数
# ============================================================
def train(_class_, dataset='mvtec', wafer_view=None, wafer_data_dir='./data', 
          save_dir='./checkpoints_vit_recontrast', use_wafer_encoder=False,
          pretrained_model='vit_small_patch14_dinov2.lvd142m', eval_interval=100):
    
    print_fn(_class_)
    setup_seed(111)

    total_iters = 1000
    batch_size = 4  # 518x518+DINOv2，12GB显存建议batch_size=4
    image_size = 518  # DINOv2原生尺寸
    crop_size = 518

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    if dataset == 'mvtec':
        train_path = 'mvtec_anomaly_detection/' + _class_ + '/train'
        test_path = 'mvtec_anomaly_detection/' + _class_
        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
    elif dataset == 'wafer':
        data_root = Path(wafer_data_dir) / '晶圆分类数据集'
        train_data = WaferTrainDataset(data_root, _class_, wafer_view, data_transform)
        test_data = WaferDataset(data_root, _class_, wafer_view, data_transform, gt_transform)
    else:
        raise ValueError(f"未知数据集: {dataset}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=2, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    # 构建模型
    if use_wafer_encoder and HAS_WAFER_ENCODER:
        print_fn("[INFO] 使用现有的ViTEncoder")
        wafer_encoder = ViTEncoder(
            img_size=224, patch_size=16, in_channels=3,
            embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1
        )
        model, encoder, encoder_freeze = build_recontrast_vit(
            wafer_encoder=wafer_encoder, use_pretrained=False
        )
    else:
        print_fn(f"[INFO] 使用DINOv2预训练模型: {pretrained_model}")
        model, encoder, encoder_freeze = build_recontrast_vit(
            wafer_encoder=None, use_pretrained=True, pretrained_model=pretrained_model
        )

    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(model.decoder.parameters()) + list(model.bottleneck.parameters()),
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5
    )
    optimizer2 = torch.optim.AdamW(
        list(model.encoder.parameters()),
        lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5
    )

    print_fn(f'train image number: {len(train_data)}')
    print_fn(f'test image number: {len(test_data)}')

    # 保存目录
    model_save_dir = Path(save_dir)
    if dataset == 'wafer' and wafer_view:
        subdir = f"{_class_}_{wafer_view}".replace(' ', '_')
        model_save_dir = model_save_dir / subdir
    elif dataset == 'mvtec':
        model_save_dir = model_save_dir / _class_
    os.makedirs(model_save_dir, exist_ok=True)

    # TensorBoard
    log_dir = os.path.join(model_save_dir, 'tensorboard', _class_)
    writer = SummaryWriter(log_dir=log_dir)

    auroc_sp_best = 0
    best_cm, best_acc, best_f1, best_thr = None, 0, 0, 0
    it = 0

    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train(encoder_bn_train=False)

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            # CutPaste增强训练分支：随机对部分图像切块重贴，制造"伪异常"
            img_aug = torch.stack([cut_paste(i) if np.random.random() < 0.5 else i for i in img])
            en, de = model(img, img_aug)

            # 交叉重建损失（与原版ReContrast一致）
            # en = [freeze_0, freeze_1, freeze_2, freeze_3, train_0, train_1, train_2, train_3]
            # de = [train_recon_0, train_recon_1, train_recon_2, train_recon_3,
            #       freeze_recon_0, freeze_recon_1, freeze_recon_2, freeze_recon_3]
            # 前半: freeze vs train_recon (训练分支预测冻结特征)
            # 后半: train vs freeze_recon (冻结分支预测训练特征)
            n = len(en) // 2  # 4
            alpha = min(-3 + 4 * it / (total_iters * 0.1), 1.0)
            loss = (global_cosine_hm_tokens(en[:n], de[:n], alpha=alpha) / 2 +
                    global_cosine_hm_tokens(en[n:], de[n:], alpha=alpha) / 2)

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), it)

            if (it + 1) % eval_interval == 0:
                _, auroc_sp, _, cm, acc, f1, thr = evaluate_full(
                    model, test_dataloader, device, _class_=_class_)
                model.train(encoder_bn_train=False)

                writer.add_scalar('Metrics/AUROC', auroc_sp, it)
                writer.add_scalar('Metrics/Accuracy', acc, it)
                writer.add_scalar('Metrics/F1', f1, it)

                if cm is not None and len(cm) == 2:
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                    fnr = fn / (tp + fn + 1e-8)
                    fpr = fp / (fp + tn + 1e-8)
                    writer.add_scalar('Metrics/FNR', fnr, it)
                    writer.add_scalar('Metrics/FPR', fpr, it)

                print_fn(f'Sample Auroc:{auroc_sp:.3f} | Acc:{acc:.3f} F1:{f1:.3f}')
                if cm is not None and len(cm) == 2:
                    print_fn(f'  混淆矩阵: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]')
                    print_fn(f'  漏检率(FNR):{fnr:.4f}  误检率(FPR):{fpr:.4f}')

                if auroc_sp >= auroc_sp_best:
                    auroc_sp_best = auroc_sp
                    best_cm, best_acc, best_f1, best_thr = cm, acc, f1, thr
                    torch.save({
                        'iter': it + 1,
                        'model_state_dict': model.state_dict(),
                        'auroc_sp': auroc_sp,
                        'class': _class_,
                    }, os.path.join(model_save_dir, f'best_model_{_class_}.pth'))
                    print_fn(f'  >>> 保存最优模型 (AUROC={auroc_sp:.3f})')
            it += 1
            if it == total_iters:
                break
        print_fn(f'iter [{it}/{total_iters}], loss:{np.mean(loss_list):.4f}')

    # 最终结果
    print_fn(f'\n{"=" * 60}')
    print_fn(f'  {_class_} 最终结果 (最优模型)')
    print_fn(f'{"=" * 60}')
    print_fn(f'  Sample AUROC: {auroc_sp_best:.4f}')
    print_fn(f'  Accuracy:     {best_acc:.4f}')
    print_fn(f'  F1 Score:     {best_f1:.4f}')
    print_fn(f'  Threshold:    {best_thr:.4f}')
    fnr, fpr = None, None
    if best_cm is not None:
        tn, fp, fn, tp = best_cm[0][0], best_cm[0][1], best_cm[1][0], best_cm[1][1]
        print_fn(f'  混淆矩阵:')
        print_fn(f'    TP={tp}  FP={fp}')
        print_fn(f'    FN={fn}  TN={tn}')
        fnr = fn / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        print_fn(f'    漏检率(FNR): {fnr:.4f}  误检率(FPR): {fpr:.4f}')

    # 加载最优模型，保存混淆矩阵分类图片
    best_model_path = os.path.join(model_save_dir, f'best_model_{_class_}.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        _, _, _, _, _, _, _, img_paths, gt_list, _, preds = evaluate_full(
            model, test_dataloader, device, _class_=_class_, return_details=True)
        confusion_save_root = os.path.join(model_save_dir, 'confusion_images')
        cat_counts = save_confusion_images(img_paths, gt_list, preds,
                                           confusion_save_root, _class_)
        print_fn(f'  混淆矩阵图片已保存至: {confusion_save_root}/{_class_}/')
        for cat_name in ['TP', 'FP', 'FN', 'TN']:
            print_fn(f'    {cat_name}: {len(cat_counts[cat_name])} 张')
        print_fn(f'\n  -------- 错误分类图片 --------')
        for cat_name in ['FP', 'FN']:
            if cat_counts[cat_name]:
                print_fn(f'  [{cat_name}] 共 {len(cat_counts[cat_name])} 张:')
                for p in cat_counts[cat_name]:
                    print_fn(f'    {os.path.basename(p)}')

    writer.close()
    return 0, auroc_sp_best, 0, best_acc, best_f1, fnr, fpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReContrast ViT + DINOv2 训练评估')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'wafer'],
                        help='数据集: mvtec 或 wafer')
    parser.add_argument('--wafer_data_dir', type=str, default='./data',
                        help='晶圆数据根目录')
    parser.add_argument('--wafer_category', type=str, default=None,
                        help='晶圆品类 (指定单个品类时用)')
    parser.add_argument('--wafer_view', type=str, default='ALL',
                        choices=['ALL', 'UP', 'DOWN'],
                        help='晶圆视图: ALL/UP/DOWN')
    parser.add_argument('--categories', type=str, default=None,
                        help='要训练的品类列表，逗号分隔')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_vit_recontrast',
                        help='模型和结果保存目录')
    parser.add_argument('--save_name', type=str, default='recontrast_vit',
                        help='实验命名')
    parser.add_argument('--use_wafer_encoder', action='store_true',
                        help='使用现有的ViTEncoder而不是DINOv2')
    parser.add_argument('--pretrained_model', type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='DINOv2预训练模型名称')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='评估间隔（iters），默认100')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    args = parser.parse_args()

    # 确定品类列表
    item_list = []
    if args.dataset == 'mvtec':
        if args.categories:
            item_list = [c.strip() for c in args.categories.split(',')]
        else:
            item_list = ['grid', 'tile', 'screw']
    elif args.dataset == 'wafer':
        if args.wafer_category:
            item_list = [args.wafer_category]
        elif args.categories:
            item_list = [c.strip() for c in args.categories.split(',')]
        else:
            data_root = Path(args.wafer_data_dir) / '晶圆分类数据集'
            item_list = sorted([
                d.name for d in data_root.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ])

    print_fn = None
    print(f'>>> 数据集: {args.dataset}, 品类: {item_list}')
    if args.dataset == 'wafer':
        print(f'>>> 视图: {args.wafer_view}')
    print(f'>>> 使用模型: {"现有ViTEncoder" if args.use_wafer_encoder else args.pretrained_model}')

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # CUDA设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f'[INFO] 使用 CUDA: {torch.cuda.get_device_name(int(args.gpu))}')
    else:
        device = torch.device('cpu')
        print('[WARNING] CUDA不可用，回退到CPU')
    print_fn(str(device))

    result_list = []
    for item in item_list:
        auroc_px_best, auroc_sp_best, aupro_px_best, acc, f1, fnr, fpr = train(
            item, dataset=args.dataset,
            wafer_view=args.wafer_view, wafer_data_dir=args.wafer_data_dir,
            save_dir=args.save_dir, use_wafer_encoder=args.use_wafer_encoder,
            pretrained_model=args.pretrained_model, eval_interval=args.eval_interval)
        result_list.append([item, auroc_sp_best, acc, f1, fnr, fpr])

    # 汇总
    print_fn(f'\n{"=" * 60}')
    print_fn(f'  汇总结果')
    print_fn(f'{"=" * 60}')
    print_fn(f'  {"品类":<16} {"Sample AUROC":>12} {"Acc":>8} {"F1":>8} {"FPR":>8} {"FNR":>8}')
    for r in result_list:
        fpr_str = f'{r[4]:>8.4f}' if r[4] is not None else f'{"N/A":>8}'
        fnr_str = f'{r[5]:>8.4f}' if r[5] is not None else f'{"N/A":>8}'
        print_fn(f'  {r[0]:<16} {r[1]:>12.4f} {r[2]:>8.4f} {r[3]:>8.4f} {fpr_str} {fnr_str}')

    avg_auroc = np.mean([r[1] for r in result_list])
    avg_acc = np.mean([r[2] for r in result_list])
    avg_f1 = np.mean([r[3] for r in result_list])
    avg_fpr = np.mean([r[4] for r in result_list if r[4] is not None])
    avg_fnr = np.mean([r[5] for r in result_list if r[5] is not None])
    print_fn(f'  {"平均":<16} {avg_auroc:>12.4f} {avg_acc:>8.4f} {avg_f1:>8.4f} {avg_fpr:>8.4f} {avg_fnr:>8.4f}')
