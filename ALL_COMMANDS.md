# 🟡 毕业设计 - 所有数据集 / 消融实验命令清单

> 工作目录：`C:\Users\21196\Desktop\毕业设计\code`（即本文件所在目录）
> 所有命令在命令行中从 `code` 目录执行

---

## 📂 目录结构速览

```
code/
├── wafer_defect_detection/train.py ← ViT+MoCo V3 主模型（模块化版本）
├── train_simsiam.py ← DenseSimSiam 对比方案
├── run_ablation.py ← 消融实验（6个实验逐步添加模块）
├── baseline/train.py ← 早期 Baseline 版本
├── download_mvtec.py ← 下载 MVTec AD 数据集
├── data/数据集/数据集/ ← 晶圆数据集（10种产品类型）
├── mvtec_anomaly_detection/ ← MVTec AD 数据集
├── checkpoints_v3/ ← ViT+MoCo 模型保存
├── checkpoints_simsiam/ ← DenseSimSiam 模型保存
└── ablation_results/ ← 消融实验输出
```

---

## 🏭 可用数据集

### 晶圆数据集（半导体产品质量检测）
| 产品类型 | 目录名 |
|---------|--------|
| BGA 12x4 | data/数据集/数据集/BGA 12x4 |
| BGA S5E 16x7 | data/数据集/数据集/BGA S5E 16x7 |
| ESSD 12x4 | data/数据集/数据集/ESSD 12x4 |
| ESSD 12x5 | data/数据集/数据集/ESSD 12x5 |
| INAND 16x5 | data/数据集/数据集/INAND 16x5 |
| INAND 19x5 | data/数据集/数据集/INAND 19x5 |
| MicroSD 20x4 | data/数据集/数据集/MicroSD 20x4 |
| SDSIP 22x3 | data/数据集/数据集/SDSIP 22x3 |
| UBGA 12x5 | data/数据集/数据集/UBGA 12x5 |
| Defect sample | data/数据集/数据集/Defect sample |

### MVTec AD 标准数据集（15个类别）
```
bottle, cable, capsule, carpet, grid,
hazelnut, leather, metal_nut, pill, screw,
tile, toothbrush, transistor, wood, zipper
```

---

## 🔬 一、ViT + MoCo V3（主模型）

> 入口脚本：`python -m wafer_defect_detection.train`（模块化版本）
> ⚠️ run.bat/run.sh 中引用的 `train_improved_v3.py` 需改为 `-m wafer_defect_detection.train`

### 1.1 晶圆数据集 - 训练

```bash
# 基础训练（200 epoch，启用全部改进模块）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 快速测试（20 epoch，小batch）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 20 --batch_size 16 --lr 1e-3 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 基础版（关闭所有改进，纯ViT+MoCo）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3

# 不划分验证集（全部数据用于训练）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --val_ratio 0.0 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 自定义超参数
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 300 --batch_size 64 --lr 5e-4 --embed_dim 512 --queue_size 2048 --temperature 0.1 --hypersphere_weight 0.2 --discriminator_weight 0.1 --generator_weight 0.1 --cutpaste_prob 0.5 --seed 123 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 不同图像尺寸
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 128 --epochs 200 --batch_size 64 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 384 --epochs 200 --batch_size 16 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

### 1.2 晶圆数据集 - 评估

```bash
# 使用自动找到的最佳模型评估
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --save_dir ./checkpoints_v3

# 指定checkpoint路径评估
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_v3/best_model_wafer.pth

# 不同评分模式评估
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode combined --checkpoint ./checkpoints_v3/best_model_wafer.pth

python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode mahal --checkpoint ./checkpoints_v3/best_model_wafer.pth

python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode memory --checkpoint ./checkpoints_v3/best_model_wafer.pth

python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode max --checkpoint ./checkpoints_v3/best_model_wafer.pth
```

### 1.3 MVTec AD - 单类别训练+评估

```bash
# 训练 bottle（默认）
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 训练 cable
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# === MVTec 15 个类别全部训练命令（粘贴即用）===

# bottle
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# cable
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# capsule
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# carpet
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# grid
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# hazelnut
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# leather
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# metal_nut
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# pill
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# screw
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# tile
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# toothbrush
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# transistor
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# wood
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# zipper
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

### 1.4 MVTec AD - 单类别评估

```bash
# 评估 bottle
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# === MVTec 15 类评估命令 ===
# bottle
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# cable
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# capsule
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# carpet
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# grid
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# hazelnut
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# leather
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# metal_nut
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# pill
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# screw
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# tile
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# toothbrush
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# transistor
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# wood
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined

# zipper
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
```

### 1.5 MVTec AD - 一键训练+评估所有15类

```bash
# 自动遍历所有15个类别，训练→评估→汇总
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 快速版（50epoch）
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 50 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

---

## 🔬 二、DenseSimSiam（对比方案）

> 入口脚本：`python train_simsiam.py`
> SimSiam 无动量编码器+无负样本队列，节省50%显存

### 2.1 晶圆数据集 - 训练

```bash
# 完整训练
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

# 基础版（关闭辅助模块）
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32

# 仅全局 SimSiam（无多尺度、无稠密）
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_multiscale --use_dense

# 仅稠密 SimSiam（无多尺度）
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_dense

# 仅多尺度 SimSiam（无稠密）
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_multiscale

# 自定义损失权重
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --global_weight 1.0 --dense_weight 0.5 --multiscale_weight 0.5 --hypersphere_weight 0.2 --discriminator_weight 0.1 --generator_weight 0.1 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

# 自定义模型维度
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --embed_dim 512 --proj_dim 256 --pred_hidden 128 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

# 不同图像尺寸
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --img_size 128 --epochs 200 --batch_size 64 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 2.2 晶圆数据集 - 评估

```bash
# 自动找checkpoint
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --save_dir ./checkpoints_simsiam --score_mode combined

# 指定checkpoint
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode combined

# 不同评分模式
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode mahal
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode memory
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode max
```

### 2.3 晶圆数据集 - 训练+评估一条龙

```bash
python train_simsiam.py --mode all --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 2.4 MVTec AD - 单类别训练

```bash
# === DenseSimSiam MVTec 15类训练 ===
python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 2.5 MVTec AD - 单类别评估

```bash
# bottle 评估
python train_simsiam.py --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --save_dir ./checkpoints_simsiam --score_mode combined
```

### 2.6 MVTec AD - 一键训练+评估所有15类

```bash
# DenseSimSiam 全类别
python train_simsiam.py --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

---

## 🧪 三、消融实验（run_ablation.py）

> 入口脚本：`python run_ablation.py`
> 6个实验逐步添加模块，验证各模块贡献

### 实验设计
| 实验 | 描述 | 增量模块 |
|------|------|---------|
| Exp0 | ViT+MoCo（纯baseline） | — |
| Exp1 | + 多尺度特征融合 | 多尺度 |
| Exp2 | + CutPaste合成异常 | CutPaste |
| Exp3 | + 超球面约束(CFA) | CFA |
| Exp4 | + 特征生成-判别(SimpleNet) | SimpleNet |
| Exp5 | + 记忆库(PatchCore) = 完整版 | 记忆库 |

### 3.1 运行全部6个实验（默认）

```bash
# 默认配置
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32

# 快速消融（20epoch，测试用）
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 20 --batch_size 32
```

### 3.2 只跑指定范围

```bash
# 只跑前3个实验 (Exp0, Exp1, Exp2)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "0-2"

# 只跑单个实验 (Exp0 baseline)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "0"

# 跑指定实验 (Exp3 和 Exp5)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "3,5"
```

### 3.3 自定义消融参数

```bash
# 自定义超参数
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --img_size 224 --embed_dim 384 --queue_size 1024 --momentum 0.999 --temperature 0.07 --epochs 200 --batch_size 32 --lr 1e-3 --num_workers 4 --seed 42 --cutpaste_prob 0.3

# 大尺寸模型消融
python run_ablation.py --data_dir ./data --save_dir ./ablation_results_large --img_size 384 --embed_dim 512 --epochs 200 --batch_size 16 --lr 5e-4
```

---

## 🏗️ 四、Baseline（早期版本）

> 入口脚本：`python baseline/train.py`
> 早期版本 ViT+MoCo，无模块化

```bash
python baseline/train.py
```

---

## 📥 五、下载数据集

```bash
# 下载 MVTec AD 数据集（需要网络）
python download_mvtec.py

# 或用shell脚本
bash download_mvtec.sh
```

---

## 📊 六、常用组合场景

### 场景A：论文核心实验（晶圆数据集 + 消融实验）

```bash
# Step 1: 消融实验（验证各模块贡献）
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32

# Step 2: 完整版训练
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# Step 3: DenseSimSiam 对比
python train_simsiam.py --mode all --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 场景B：MVTec AD 标准数据集验证

```bash
# ViT+MoCo 全15类
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# DenseSimSiam 全15类
python train_simsiam.py --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 场景C：快速验证（少量epoch，确认代码能跑通）

```bash
# ViT+MoCo 快速测试
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 10 --batch_size 16 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 消融快速测试
python run_ablation.py --data_dir ./data --save_dir ./ablation_results_quick --epochs 10 --batch_size 16

# SimSiam 快速测试
python train_simsiam.py --mode all --dataset wafer --data_dir ./data --epochs 10 --batch_size 16 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

---

## ⚙️ 七、完整参数速查

### ViT+MoCo V3 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | train | train / eval / train_eval_all |
| `--dataset` | wafer | wafer / mvtec |
| `--data_dir` | ./data | 晶圆数据集根目录 |
| `--mvtec_dir` | ./mvtec_anomaly_detection | MVTec路径 |
| `--mvtec_category` | bottle | MVTec类别名 |
| `--img_size` | 224 | 输入图片尺寸 |
| `--embed_dim` | 384 | ViT嵌入维度 |
| `--queue_size` | 1024 | MoCo队列大小 |
| `--momentum` | 0.999 | 动量更新系数 |
| `--temperature` | 0.07 | 对比损失温度 |
| `--use_multiscale` | True | 多尺度特征 |
| `--use_cutpaste` | True | CutPaste增强 |
| `--use_feature_generator` | True | SimpleNet特征生成器 |
| `--use_hypersphere` | True | CFA超球面约束 |
| `--hypersphere_weight` | 0.1 | 超球面损失权重 |
| `--discriminator_weight` | 0.05 | 判别器损失权重 |
| `--generator_weight` | 0.05 | 生成器损失权重 |
| `--cutpaste_prob` | 0.3 | CutPaste概率 |
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--num_workers` | 4 | 数据加载线程 |
| `--seed` | 42 | 随机种子 |
| `--val_ratio` | 0.2 | 验证集比例 |
| `--save_dir` | ./checkpoints_v3 | 模型保存目录 |
| `--checkpoint` | — | 评估时用的checkpoint路径 |
| `--pca_components` | None | PCA降维维度 |
| `--score_mode` | combined | combined/mahal/memory/max |

### DenseSimSiam 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | train | train/eval/all/train_eval_all |
| `--dataset` | wafer | wafer/mvtec |
| `--use_dense` | True | 稠密(patch级)SimSiam |
| `--use_multiscale` | True | 多尺度SimSiam |
| `--global_weight` | 1.0 | 全局SimSiam权重 |
| `--dense_weight` | 0.3 | 稠密SimSiam权重 |
| `--multiscale_weight` | 0.3 | 多尺度SimSiam权重 |
| `--proj_dim` | 256 | SimSiam投影维度 |
| `--pred_hidden` | 128 | SimSiam预测器隐藏维度 |

### 消融实验参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_range` | None | 指定实验范围，如 "0-2" 或 "3,5" |
| `--save_dir` | ./ablation_results | 结果保存目录 |

---

|---
| > 🟡🦖 奶龙整理完毕！所有命令都是直接复制粘贴就能跑哒～

## 🔄 七、ReContrast（对比方案二）

> 入口脚本：`python recontrast_wafer.py`
> 基于 ResNet + WideResNet 的 ReContrast 重构对比方法
> 依赖包：`recontrast/` (dataset.py + utils.py + models/)

### 7.1 MVTec AD - 训练+评估

```bash
# 单类别（晶圆相似类）
python recontrast_wafer.py --dataset mvtec --categories "grid,tile,screw"

# 指定任意MVTec类别
python recontrast_wafer.py --dataset mvtec --categories "bottle,capsule,carpet"

# 全部MVTec 15类（用晶圆相似默认）
python recontrast_wafer.py --dataset mvtec
```

### 7.2 晶圆数据集 - 训练+评估

```bash
# 单品类（指定品类）
python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7"

# 单品类+指定视图
python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

# 多品类逗号分隔
python recontrast_wafer.py --dataset wafer --categories "BGA S5E 16x7,ESSD 12x5"

# 全品类自动检测
python recontrast_wafer.py --dataset wafer
```

### 7.3 ReContrast 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec | mvtec / wafer |
| `--categories` | grid,tile,screw | 品类列表，逗号分隔 |
| `--wafer_category` | None | 单个晶圆品类 |
| `--wafer_view` | ALL | ALL / UP / DOWN |
| `--wafer_data_dir` | ./data | 晶圆数据根目录 |
| `--save_dir` | ./saved_results | 结果保存目录 |
| `--save_name` | recontrast_wafer | 实验命名 |
| `--gpu` | 0 | GPU ID |
| `total_iters` | 2000 | 训练总迭代数（硬编码） |

---

## ⚙️ 八、完整参数速查
