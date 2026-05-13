"""
ReContrast — 支持 MVTec AD 和晶圆数据集 (CUDA版)
用法:
  # MVTec
  python recontrast_wafer.py --dataset mvtec --categories bottle,capsule

  # 晶圆单个品类
  python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

  # 晶圆多个品类
  python recontrast_wafer.py --dataset wafer --categories "BGA S5E 16x7,ESSD 12x5"
"""

import torch
from recontrast.dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import glob
from torch.utils.data import DataLoader
from recontrast.models.resnet import wide_resnet50_2
from recontrast.models.de_resnet import de_wide_resnet50_2
from recontrast.models.recontrast import ReContrast
import argparse
from recontrast.utils import (setup_seed, get_device, evaluation, visualize,
                               global_cosine, global_cosine_hm, cal_anomaly_map,
                               return_best_thr, min_max_norm, cvt2heatmap, show_cam_on_image,
                               compute_pro)
from torch.nn import functional as F
from functools import partial
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image
from pathlib import Path
import warnings
import copy
import logging

warnings.filterwarnings("ignore")


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
        # ReContrast训练需要 (img, label) 格式
        return img, 0  # 所有训练样本label=0（正常）


# ============================================================
# 评估函数
# ============================================================
def evaluate_full(model, dataloader, device, _class_=None, return_details=False):
    """完整评估：AUROC + 混淆矩阵 + 准确率 + F1"""
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    img_paths = []

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)
            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
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
def train(_class_, dataset='mvtec', wafer_view=None, wafer_data_dir='./data'):
    print_fn(_class_)
    setup_seed(111)

    total_iters = 1000
    batch_size = 8
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    if dataset == 'mvtec':
        # MVTec AD 模式
        train_path = 'mvtec_anomaly_detection/' + _class_ + '/train'
        test_path = 'mvtec_anomaly_detection/' + _class_
        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
    elif dataset == 'wafer':
        # 晶圆分类模式
        data_root = Path(wafer_data_dir) / '晶圆分类数据集'
        train_data = WaferTrainDataset(data_root, _class_, wafer_view, data_transform)
        test_data = WaferDataset(data_root, _class_, wafer_view, data_transform, gt_transform)
    else:
        raise ValueError(f"未知数据集: {dataset}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=2, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    # 模型
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze,
                       bottleneck=bn, decoder=decoder)

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)

    print_fn(f'train image number: {len(train_data)}')
    print_fn(f'test image number: {len(test_data)}')

    # 保存目录
    model_save_dir = '/data/coding/under_graduate/recontrast/checkpoint'
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
            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = (global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 +
                    global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2)

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), it)

            if (it + 1) % 250 == 0:
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

                print_fn(
                    f'Sample Auroc:{auroc_sp:.3f} | Acc:{acc:.3f} F1:{f1:.3f}')
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
    fnr = None
    fpr = None
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

    writer.close()
    return 0, auroc_sp_best, 0, best_acc, best_f1, fnr, fpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReContrast — MVTec+晶圆训练评估')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'wafer'],
                        help='数据集: mvtec 或 wafer')
    parser.add_argument('--wafer_data_dir', type=str, default='./data',
                        help='晶圆数据根目录 (默认 ./data)')
    parser.add_argument('--wafer_category', type=str, default=None,
                        help='晶圆品类 (指定单个品类时用)')
    parser.add_argument('--wafer_view', type=str, default='ALL',
                        choices=['ALL', 'UP', 'DOWN'],
                        help='晶圆视图: ALL/UP/DOWN')
    parser.add_argument('--categories', type=str, default=None,
                        help='要训练的品类列表，逗号分隔 (如 "grid,tile" 或 "BGA S5E 16x7,ESSD 12x5")')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='recontrast_wafer')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    args = parser.parse_args()

    # 确定品类列表
    item_list = []
    if args.dataset == 'mvtec':
        if args.categories:
            item_list = [c.strip() for c in args.categories.split(',')]
        else:
            # 默认：晶圆相似的3个MVTec类别
            item_list = ['grid', 'tile', 'screw']
    elif args.dataset == 'wafer':
        if args.wafer_category:
            item_list = [args.wafer_category]
        elif args.categories:
            item_list = [c.strip() for c in args.categories.split(',')]
        else:
            # 默认：自动检测所有品类
            data_root = Path(args.wafer_data_dir) / '晶圆分类数据集'
            item_list = sorted([
                d.name for d in data_root.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ])

    print_fn = None
    print(f'>>> 数据集: {args.dataset}, 品类: {item_list}')
    if args.dataset == 'wafer':
        print(f'>>> 视图: {args.wafer_view}')

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

    # 导入MVTecDataset（仅用于MVTec模式）
    if args.dataset == 'mvtec':
        from recontrast.dataset import MVTecDataset

    result_list = []
    for item in item_list:
        auroc_px_best, auroc_sp_best, aupro_px_best, acc, f1, fnr, fpr = train(
            item, dataset=args.dataset,
            wafer_view=args.wafer_view, wafer_data_dir=args.wafer_data_dir)
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
