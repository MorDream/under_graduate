# 对比实验结果（表4-X）

数据集：3个品类
指标：图像级 AUROC
生成时间：2026-05-14 19:29:48

| 品类 | ViT+ReContrast (本文) |
|------|--------|
| BGA S5E 16x7 | **0.9350** |
| ESSD 12x5 | — |
| INAND 19x5 | — |
|------|--------|
| **平均** | 0.9350 |

> 注：PaDiM/PatchCore 数据引用自原论文。ViT+ReContrast 使用 DINOv2 + CutPaste + 交叉重建。