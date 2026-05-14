#!/usr/bin/env python3
"""
4.5 可视化 — 论文图表生成脚本
===============================
论文第四章第五节所需图表：异常热力图、训练曲线、模型架构图

功能：
  1. 异常热力图 — 对正常/缺陷样本生成热力图对比
  2. 训练曲线图 — 从 TensorBoard 提取 Loss/AUROC 并绘图
  3. 模型架构图 — 生成 ViT+ReContrast 架构示意图

用法：
  # 生成全部可视化（需要先有训练好的模型）
  python 生成实验数据/4.5_可视化/generate_figures.py --checkpoint <路径> --categories carpet --all

  # 只生成热力图
  python 生成实验数据/4.5_可视化/generate_figures.py --checkpoint <路径> --categories carpet --figs heatmap

  # 只提取训练曲线（从TensorBoard）
  python 生成实验数据/4.5_可视化/generate_figures.py --tensorboard_dir <路径> --figs curves

  # 只生成架构图
  python 生成实验数据/4.5_可视化/generate_figures.py --figs architecture

输出：
  生成实验数据/4.5_可视化/output/
  ├── heatmaps/          # 热力图：正常vs缺陷，多类别
  ├── curves/            # 训练曲线：loss + AUROC epoch图
  └── architecture/      # 架构示意图
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── 输出目录 ───
OUTPUT_DIR = Path('./生成实验数据/4.5_可视化/output')


# ═══════════════════════════════════════════════
# 1. 异常热力图生成
# ═══════════════════════════════════════════════
def generate_heatmaps(checkpoint_path, categories, dataset='mvtec', data_dir='./data',
                      mvtec_dir='./mvtec_anomaly_detection', num_samples=4):
    """
    使用训练好的 ViT+ReContrast 模型，对每个品类生成正常/缺陷样本的热力图
    """
    print(f"\n{'='*70}")
    print(f"  1. 生成异常热力图")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  品类: {categories}")
    print(f"{'='*70}")
    
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    import cv2
    
    # 导入项目模块
    sys.path.insert(0, str(Path(checkpoint_path).parent.parent.parent))
    from recontrast.models.recontrast_vit import build_recontrast_vit, load_dinov2_encoder
    from recontrast.dataset import MVTecDataset
    from recontrast.utils import cal_anomaly_map, min_max_norm, cvt2heatmap, show_cam_on_image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    heatmap_dir = OUTPUT_DIR / 'heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    for cat in categories:
        print(f"\n  📦 {cat}")
        cat_dir = heatmap_dir / cat.replace(' ', '_')
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 构建模型
            encoder, encoder_dim = load_dinov2_encoder(device)
            model = build_recontrast_vit(encoder, encoder_dim).to(device)
            
            # 加载checkpoint（支持多品类目录结构）
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.is_dir():
                # 在目录下找 .pth 文件
                pth_files = list(ckpt_path.glob('*.pth'))
                if not pth_files:
                    print(f"    ⚠ 未找到 .pth 文件，跳过")
                    continue
                ckpt_file = pth_files[0]
            else:
                ckpt_file = ckpt_path
            
            state = torch.load(str(ckpt_file), map_location=device, weights_only=False)
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state, strict=False)
            model.eval()
            
            # 加载测试数据
            test_transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            test_data = MVTecDataset(
                root=mvtec_dir, category=cat, transform=test_transform,
                gt_transform=transforms.Resize((518, 518)), phase='test'
            )
            
            # 分别收集正常和缺陷样本
            normal_samples = []
            defect_samples = []
            for i in range(len(test_data)):
                img, label, gt, _ = test_data[i]
                if len(normal_samples) < num_samples and label == 0:
                    normal_samples.append((img, f"normal_{len(normal_samples):02d}"))
                if len(defect_samples) < num_samples and label == 1:
                    defect_samples.append((img, f"defect_{len(defect_samples):02d}"))
                if len(normal_samples) >= num_samples and len(defect_samples) >= num_samples:
                    break
            
            # 生成热力图
            for img_tensor, name in normal_samples + defect_samples:
                img_batch = img_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    en, de = model(img_batch)
                    anomaly_map, _ = cal_anomaly_map(en, de, out_size=518, amap_mode='mul')
                
                amap = anomaly_map[0]
                amap = min_max_norm(amap)
                amap = cv2.GaussianBlur(amap, (7, 7), 4)
                
                # 反归一化原图
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                heatmap = cvt2heatmap(amap * 255)
                overlay = show_cam_on_image(img_np / 255.0, amap)
                
                # 拼接：原图 | 热力图 | 叠加图
                h, w = img_np.shape[:2]
                combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
                combined[:, :w] = img_np
                combined[:, w:2*w] = heatmap
                combined[:, 2*w:] = (overlay * 255).astype(np.uint8)
                
                save_path = cat_dir / f'{cat.replace(" ","_")}_{name}.jpg'
                cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print(f"    ✅ {save_path.name}")
            
            print(f"    📁 保存至: {cat_dir}")
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    print(f"\n  热力图保存至: {heatmap_dir}")


# ═══════════════════════════════════════════════
# 2. 训练曲线提取
# ═══════════════════════════════════════════════
def extract_training_curves(tensorboard_dir, categories=None):
    """
    从 TensorBoard event 文件中提取 Loss / AUROC 训练曲线并绘图
    """
    print(f"\n{'='*70}")
    print(f"  2. 提取训练曲线")
    print(f"  TensorBoard: {tensorboard_dir}")
    print(f"{'='*70}")
    
    curves_dir = OUTPUT_DIR / 'curves'
    curves_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  ⚠ tensorboard 未安装: pip install tensorboard")
        return
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    tb_path = Path(tensorboard_dir)
    if not tb_path.exists():
        print(f"  ❌ 路径不存在: {tensorboard_dir}")
        return
    
    # 收集所有 event 文件
    event_files = list(tb_path.rglob('events.out.tfevents.*'))
    if not event_files:
        print(f"  ❌ 未找到 event 文件")
        return
    
    # 按子目录分组
    from collections import defaultdict
    groups = defaultdict(lambda: {'loss': [], 'auroc': []})
    
    for ef in event_files:
        subdir = ef.parent.name
        try:
            ea = EventAccumulator(str(ef.parent))
            ea.Reload()
            
            # 提取标量
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                if 'loss' in tag.lower() or 'total' in tag.lower():
                    groups[subdir]['loss'].append((tag, steps, values))
                if 'auroc' in tag.lower():
                    groups[subdir]['auroc'].append((tag, steps, values))
        except Exception as e:
            print(f"    ⚠ {ef.parent.name}: {e}")
    
    # 绘图
    for subdir, data in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Training Curves — {subdir}', fontsize=14)
        
        # Loss 曲线
        ax = axes[0]
        for tag, steps, values in data['loss']:
            ax.plot(steps, values, label=tag, linewidth=1.5, alpha=0.8)
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # AUROC 曲线
        ax = axes[1]
        for tag, steps, values in data['auroc']:
            ax.plot(steps, values, label=tag, linewidth=1.5, alpha=0.8)
        ax.set_title('AUROC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        save_path = curves_dir / f'training_curves_{subdir}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✅ {save_path}")
    
    print(f"\n  曲线保存至: {curves_dir}")


# ═══════════════════════════════════════════════
# 3. 模型架构图
# ═══════════════════════════════════════════════
def generate_architecture_diagram():
    """
    生成 ViT+ReContrast 架构示意图（ASCII + PNG）
    """
    print(f"\n{'='*70}")
    print(f"  3. 生成模型架构图")
    print(f"{'='*70}")
    
    arch_dir = OUTPUT_DIR / 'architecture'
    arch_dir.mkdir(parents=True, exist_ok=True)
    
    # ASCII 版本
    ascii_art = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ViT + ReContrast 异常检测架构                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   正常图像 x                    CutPaste增强 x_aug            ║
    ║   ┌──────────┐                 ┌──────────────┐             ║
    ║   │  Input   │                 │  Input(Aug)  │             ║
    ║   │ 518×518  │                 │   518×518    │             ║
    ║   └────┬─────┘                 └──────┬───────┘             ║
    ║        │                              │                      ║
    ║   ┌────▼─────────────────────┐  ┌────▼─────────────────────┐ ║
    ║   │  ViT-Small (DINOv2)      │  │  ViT-Small (DINOv2)      │ ║
    ║   │  ❄️ FROZEN 教师分支       │  │  🔥 TRAIN 学生分支        │ ║
    ║   │  Patch Embedding          │  │  Patch Embedding          │ ║
    ║   │  ↓                        │  │  ↓                        │ ║
    ║   │  Transformer ×12          │  │  Transformer ×12          │ ║
    ║   │  ↓                        │  │  ↓                        │ ║
    ║   │  多尺度特征 {2,4,6,8}层   │  │  多尺度特征 {2,4,6,8}层   │ ║
    ║   └────┬──────────────────────┘  └────┬──────────────────────┘ ║
    ║        │                              │                      ║
    ║        │   ┌──────────────────────────┘                      ║
    ║        │   │                                                  ║
    ║   ┌────▼───▼─────────────────────────────────────────────┐   ║
    ║   │            BottleneckViT (n_layers=4)                 │   ║
    ║   │   交叉重建: freeze_feat → 预测 train_feat              │   ║
    ║   │            train_feat  → 预测 freeze_feat             │   ║
    ║   │   + 难例挖掘 (α自适应调度)                              │   ║
    ║   └────────────────────────┬─────────────────────────────┘   ║
    ║                            │                                  ║
    ║   ┌────────────────────────▼─────────────────────────────┐   ║
    ║   │            DecoderViT                                │   ║
    ║   │   MLP 重建头: 384D → ... → 384D                       │   ║
    ║   │   Token级 CosineDistance 计算异常分数                  │   ║
    ║   └────────────────────────┬─────────────────────────────┘   ║
    ║                            │                                  ║
    ║   ┌────────────────────────▼─────────────────────────────┐   ║
    ║   │           异常分数图 (Anomaly Map)                    │   ║
    ║   │   上采样 → 518×518 → 高斯平滑 → 热力图                │   ║
    ║   └──────────────────────────────────────────────────────┘   ║
    ║                                                              ║
    ║   损失函数: L = L_cross_recon + L_hard_mining                ║
    ║   L_cross_recon: Token级余弦距离 (交叉重建)                   ║
    ║   L_hard_mining: 仅对重建差的token保留梯度 (α控制)            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    txt_path = arch_dir / 'architecture_ascii.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(ascii_art)
    print(f"    ✅ {txt_path}")
    
    # 同时也生成一个包含架构描述的 JSON（方便后续画矢量图）
    arch_desc = {
        'title': 'ViT + ReContrast 架构',
        'components': [
            {'name': 'Input', 'detail': '原始图像 + CutPaste增强图像, 518×518'},
            {'name': 'Teacher Encoder', 'detail': 'ViT-Small (DINOv2 预训练, ❄冻结)', 
             'output': '多尺度 patch tokens (层2/4/6/8)'},
            {'name': 'Student Encoder', 'detail': 'ViT-Small (DINOv2 预训练, 🔥可训练)',
             'output': '多尺度 patch tokens (层2/4/6/8)'},
            {'name': 'BottleneckViT', 'detail': '交叉重建 + 难例挖掘, n_layers=4',
             'detail_cn': '教师→学生交叉特征预测, 反向亦然'},
            {'name': 'DecoderViT', 'detail': 'MLP重建头, 输出重建后的token特征'},
            {'name': 'Anomaly Detection', 'detail': 'Token级 CosineDistance → 上采样 → 高斯平滑 → 热力图'},
        ],
        'loss_functions': [
            {'name': 'L_cross_recon', 'formula': '1 - cos(freeze_feat, de_train_feat) + 1 - cos(train_feat, de_freeze_feat)'},
            {'name': 'L_hard_mining', 'formula': '仅保留 distance > mean+α*std 的token梯度, α动态从-3→1'},
        ],
    }
    json_path = arch_dir / 'architecture_desc.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(arch_desc, f, indent=2, ensure_ascii=False)
    print(f"    ✅ {json_path}")
    
    print(f"\n  架构图保存至: {arch_dir}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='4.5 可视化 — 论文图表生成')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='训练好的模型checkpoint路径（用于热力图）')
    parser.add_argument('--tensorboard_dir', type=str, default='./recontrast/checkpoint/tensorboard',
                       help='TensorBoard event文件目录（用于训练曲线）')
    parser.add_argument('--categories', type=str, default='carpet',
                       help='品类名，逗号分隔')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'wafer'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--mvtec_dir', type=str, default='./mvtec_anomaly_detection')
    parser.add_argument('--figs', type=str, default='architecture',
                       help='要生成的图表: heatmap, curves, architecture, 或 all')
    parser.add_argument('--all', action='store_true', help='生成全部图表')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='每个品类每种标签的样本数（热力图用）')
    args = parser.parse_args()
    
    categories = [c.strip() for c in args.categories.split(',')]
    
    print(f"\n{'#'*70}")
    print(f"# 4.5 可视化 — 论文图表生成")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    if args.all:
        figs = ['architecture', 'curves', 'heatmap']
    else:
        figs = [f.strip() for f in args.figs.split(',')]
    
    for fig in figs:
        if fig == 'heatmap':
            if not args.checkpoint:
                print("\n  ⚠ 跳过热力图：需要 --checkpoint 参数指定模型路径")
                continue
            generate_heatmaps(args.checkpoint, categories, args.dataset,
                            args.data_dir, args.mvtec_dir, args.num_samples)
        
        elif fig == 'curves':
            extract_training_curves(args.tensorboard_dir, categories)
        
        elif fig == 'architecture':
            generate_architecture_diagram()
    
    print(f"\n{'='*70}")
    print(f"  可视化生成完成！输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
