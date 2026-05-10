# 半导体晶圆无监督缺陷检测

基于 ViT + MoCo v2 的半导体晶圆图像无监督异常检测系统。

## 项目结构

```
├── wafer_defect_detection/        # 核心模块包
│   ├── models/
│   │   ├── vit_encoder.py         # ViT 编码器
│   │   └── moco.py                # MoCo v2 对比学习框架
│   ├── detectors/
│   │   └── anomaly_detector.py    # 异常检测器 (PCA + Mahalanobis)
│   ├── data/
│   │   ├── wafer_dataset.py       # 晶圆数据集
│   │   └── mvtec_dataset.py       # MVTec AD 数据集
│   ├── utils/
│   │   ├── device.py              # 设备管理 (XPU/CUDA/CPU)
│   │   └── transforms.py         # 数据增强 (半导体专用)
│   └── train.py                   # 训练入口
├── baseline/                      # Baseline 对比版本
│   ├── train.py                   # 基线训练脚本
│   └── README.md                  # 基线说明
├── download_mvtec.py              # MVTec AD 数据集下载
├── download_mvtec.sh              # 辅助下载脚本
├── run.bat                        # Windows 启动脚本
├── run.sh                         # Linux 启动脚本
├── run_ablation.py                # 消融实验脚本
└── .gitignore
```

## 改进点

| 改进 | 来源 | 作用 |
|------|------|------|
| 多尺度特征融合 | - | 解决微小缺陷特征被淹没 |
| 半导体专用增强 | - | 保留晶圆图物理语义 |
| PCA + Ledoit-Wolf | CFA (CVPR 2023) | 稳定协方差估计 |
| CutPaste 增强 | CVPR 2021 | 合成异常样本 |
| 特征生成器 | SimpleNet (CVPR 2023) | 伪异常特征 |
| 难负样本挖掘 | - | 增强对比学习效果 |
| 温度调度 | - | 训练稳定性 |

## 环境配置

```bash
conda create -n graduate python=3.13 -y
conda activate graduate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install scikit-learn matplotlib
```

## 快速开始

```bash
# 1. 下载数据集
python download_mvtec.py

# 2. 训练 (晶圆数据集)
python -m wafer_defect_detection.train --dataset wafer --epochs 100

# 3. 训练 (MVTec AD)
python -m wafer_defect_detection.train --dataset mvtec --mvtec_category bottle --epochs 100

# 4. 评估
python -m wafer_defect_detection.train --mode eval --dataset wafer

# 5. 消融实验
python run_ablation.py
```

## 硬件

- Intel Arc B580 (XPU) / CUDA GPU / CPU 回退
