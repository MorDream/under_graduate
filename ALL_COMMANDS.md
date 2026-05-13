# 🟡 毕业设计 - 所有训练脚本命令清单

> 工作目录：`C:\Users\21196\Desktop\毕业设计\code`（即本文件所在目录）
> 所有命令在 `code` 目录下执行

---

## 📂 目录结构

```
code/
├── recontrast_wafer.py        ← ① ReContrast ResNet版（对比基线）
├── recontrast_vit_wafer.py    ← 🔥 ReContrast ViT版 + DINOv2预训练
├── wafer_defect_detection/
│   └── train.py                ← ② ViT+MoCo 主训练（每10轮自动评估）
├── train_improved_v3.py        ← ← 桥接文件，同上
├── run_ablation.py             ← ③ 消融实验（Exp0→Exp5）
├── train_simsiam.py            ← DenseSimSiam（备用对比方案）
├── recontrast/                 ← ReContrast 模块
├── data/晶圆分类数据集/         ← 晶圆数据集（8产品族×UP/DOWN）
├── mvtec_anomaly_detection/    ← MVTec AD 数据集
├── checkpoints_v3_baseline/    ← 所有模型保存总目录
│   ├── mycode/                  ← ② ViT+MoCo 主模型（每品类子文件夹）
│   ├── recontrast/              ← ① ReContrast 模型（每品类子文件夹）
│   └── ablation/                ← ③ 消融实验（Exp0→Exp5子文件夹）
```

---

## 🏭 可用数据集

### 晶圆数据集（8个产品族）
| 产品族 | 目录名 |
|--------|--------|
| BGA 12x4 | `data/晶圆分类数据集/BGA 12x4` |
| BGA S5E 16x7 | `data/晶圆分类数据集/BGA S5E 16x7` |
| ESSD 12x4 | `data/晶圆分类数据集/ESSD 12x4` |
| ESSD 12x5 | `data/晶圆分类数据集/ESSD 12x5` |
| INAND 16x5 | `data/晶圆分类数据集/INAND 16x5` |
| INAND 19x5 | `data/晶圆分类数据集/INAND 19x5` |
| MicroSD 20x4 | `data/晶圆分类数据集/MicroSD 20x4` |
| SDSIP 22x3 | `data/晶圆分类数据集/SDSIP 22x3` |

每个产品族下有 `UP/` 和 `DOWN/` 两视图，含 `train/`（正常样本）和 `test/good` + `test/defect`。

```
mvtec_anomaly_detection/    ← MVTec AD 数据集
├── bottle, cable, capsule, carpet, grid,
├── hazelnut, leather, metal_nut, pill, screw,
├── tile, toothbrush, transistor, wood, zipper
```

### 🆕 ReContrast ViT 输出目录
```
checkpoints_vit_recontrast/    ← ViT版ReContrast模型保存
├── {品类}_{视图}/              ← 每个品类+视图的子文件夹
│   ├── best_model_*.pth        ← 最优模型
│   ├── tensorboard/            ← TensorBoard日志
│   └── confusion_images/       ← 混淆矩阵分类图片
```

---

## 🔬 一、ViT + MoCo V3（主模型）

> 入口：`python -m wafer_defect_detection.train`（模块化版本）
> 桥接入口：`python train_improved_v3.py`（等价）

**亮点：训练中每10轮自动评估一次**，打印 AUROC / F1 / Acc / 混淆矩阵 / FNR+FPR，最佳 AUROC 模型自动保存。
可通过 `--eval_interval N` 自定义间隔，设 `0` 关闭。

> 🔥 **默认行为**：不指定 `--wafer_category` 时，自动遍历**所有8个晶圆品类**，每个品类训练 **UP + DOWN** 两个视图，共计16个模型！
> 每个模型保存在 `checkpoints_v3_baseline/mycode/{品类}_{视图}/` 子文件夹中。

### 1.1 晶圆数据集 - 训练

```bash
# 【推荐】全品类逐个训练（8品类 × UP/DOWN = 16个模型，每10轮自动评估）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 指定单个品类+视图训练
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --wafer_category "BGA S5E 16x7" --wafer_view UP --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 快速测试（只跑一个品类的一个视图）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 20 --batch_size 16 --wafer_category "BGA S5E 16x7" --wafer_view UP --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 纯ViT+MoCo（关闭所有改进模块）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --no-use_cutpaste --no-use_feature_generator --no-use_hypersphere

# 不划分验证集（全部数据用于训练）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --val_ratio 0.0 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 自定义超参
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 300 --batch_size 64 --lr 5e-4 --embed_dim 512 --queue_size 2048 --temperature 0.1 --hypersphere_weight 0.2 --discriminator_weight 0.1 --generator_weight 0.1 --cutpaste_prob 0.5 --seed 123 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 不同图像尺寸（128x128）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 128 --epochs 200 --batch_size 64 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 不同图像尺寸（384x384）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 384 --epochs 200 --batch_size 16 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 自定义评估间隔（每5轮评估一次）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --eval_interval 5 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

### 1.2 晶圆数据集 - 分品类+视图训练

```bash
# 指定品类 + 视图
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --wafer_category "BGA S5E 16x7" --wafer_view UP --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 全品类遍历（自动训练所有品类×UP/DOWN）
python -m wafer_defect_detection.train --mode train_all_wafer --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

### 1.3 晶圆数据集 - 评估

```bash
# 自动找最佳模型评估
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --save_dir ./checkpoints_v3_baseline/mycode

# 指定品类+视图 自动找checkpoint
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --wafer_category "BGA S5E 16x7" --wafer_view UP --save_dir ./checkpoints_v3_baseline/mycode

# 指定完整checkpoint路径
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_v3_baseline/mycode/BGA_S5E_16x7_UP/best_model_bga_s5e_16x7_up.pth --score_mode combined

# 不同评分模式
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --score_mode mahal --checkpoint ./checkpoints_v3_baseline/mycode/BGA_S5E_16x7_UP/best_model_bga_s5e_16x7_up.pth
python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --score_mode memory --checkpoint ./checkpoints_v3_baseline/mycode/BGA_S5E_16x7_UP/best_model_bga_s5e_16x7_up.pth
```

### 1.4 MVTec AD - 训练+评估

```bash
# 单类别
python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 单类别评估
python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --checkpoint ./checkpoints_v3_baseline/mycode/mvtec_bottle/best_model_mvtec.pth --score_mode combined

# 一键训练+评估所有15类
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 快速版（50epoch）
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 50 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
```

---

## 🔬 二、ReContrast（对比基线）

> 入口：`python recontrast_wafer.py`
> 基于 ResNet + WideResNet 的重构对比方法
> 依赖：`recontrast/` 模块

```bash
# MVTec AD 单类/多类
python recontrast_wafer.py --dataset mvtec --categories "grid,tile,screw"
python recontrast_wafer.py --dataset mvtec --categories "bottle,capsule,carpet"

# MVTec 默认（晶圆相似类）
python recontrast_wafer.py --dataset mvtec

# 晶圆单品类
python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7"

# 晶圆单品类+视图
python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

# 晶圆多品类
python recontrast_wafer.py --dataset wafer --categories "BGA S5E 16x7,ESSD 12x5"

# 晶圆全品类自动检测
python recontrast_wafer.py --dataset wafer
```

---

## 🔥 三、ReContrast ViT + DINOv2（新增）

> 入口：`python recontrast_vit_wafer.py`
> 基于 ViT 编码器 + DINOv2 预训练权重的重构对比方法
> 依赖：`recontrast/models/recontrast_vit.py`, `timm`

### 🎯 预训练模型选择

| 排名 | 模型名 | 参数量 | 维度 | 推荐指数 |
|:---:|------|:------:|:----:|:--------:|
| 1 | `vit_small_patch14_dinov2.lvd142m` | 22.1M | 384 ✅ | ⭐⭐⭐⭐⭐ |
| 2 | `vit_small_patch16_224.dino` | 21.7M | 384 ✅ | ⭐⭐⭐⭐ |
| 3 | `vit_small_patch16_224.augreg_in21k` | 21.7M | 384 ✅ | ⭐⭐⭐ |

**默认使用 DINOv2 Small** (`vit_small_patch14_dinov2.lvd142m`)
- 专为密集预测设计，局部特征定位强
- 异常检测/分割任务SOTA表现
- `embed_dim=384`，完美匹配现有代码

### 3.1 晶圆数据集 - 训练

```bash
# 【推荐】使用DINOv2预训练权重（默认）
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

# 全品类自动训练（8品类×UP/DOWN = 16个模型）
python recontrast_vit_wafer.py --dataset wafer

# 指定多品类
python recontrast_vit_wafer.py --dataset wafer --categories "BGA S5E 16x7,ESSD 12x5"

# 使用现有的ViTEncoder（不用DINOv2）
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --use_wafer_encoder

# 切换其他预训练模型
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --pretrained_model vit_small_patch16_224.dino

# 自定义保存目录
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --save_dir ./my_vit_checkpoints
```

### 3.2 MVTec AD - 训练

```bash
# 单类别
python recontrast_vit_wafer.py --dataset mvtec --categories bottle

# 多类别
python recontrast_vit_wafer.py --dataset mvtec --categories "bottle,capsule,carpet"

# 默认品类
python recontrast_vit_wafer.py --dataset mvtec
```

### 3.3 其他参数

```bash
# 指定GPU
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --gpu 0

# 自定义数据目录
python recontrast_vit_wafer.py --dataset wafer --wafer_data_dir ./data --wafer_category "BGA S5E 16x7"

# 自定义评估间隔（每50 iters评估一次）
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --eval_interval 50

# 较少评估（每200 iters一次，加快训练）
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP --eval_interval 200
```

---

## 🧪 四、消融实验（原三）

> 入口：`python run_ablation.py`
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

### 4.1 运行全部6个实验

```bash
# 默认配置（200 epoch）
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32

# 快速消融（20 epoch，测试用）
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 20 --batch_size 32
```

### 4.2 指定实验范围

```bash
# 前3个实验 (Exp0, Exp1, Exp2)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "0-2"

# 单个实验 (Exp0 baseline)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "0"

# 指定实验 (Exp3 + Exp5)
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32 --exp_range "3,5"
```

### 4.3 自定义参数

```bash
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --img_size 224 --embed_dim 384 --queue_size 1024 --momentum 0.999 --temperature 0.07 --epochs 200 --batch_size 32 --lr 1e-3 --num_workers 4 --seed 42 --cutpaste_prob 0.3
```

---

## 🔄 五、DenseSimSiam（备用对比方案）（原四）

> 入口：`python train_simsiam.py`
> SimSiam 无动量编码器 + 无负样本队列

```bash
# 晶圆训练
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere

# 晶圆评估
python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --save_dir ./checkpoints_simsiam --score_mode combined

# MVTec 全15类训练
python train_simsiam.py --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

---

## 📊 六、常用组合场景（原五）

### 场景A：论文核心实验（晶圆消融 + 完整版 + 对比方法）

```bash
# Step 1: 消融实验（验证各模块贡献）
python run_ablation.py --data_dir ./data --save_dir ./ablation_results --epochs 200 --batch_size 32

# Step 2: 完整版 ViT+MoCo 训练（每10轮自动评估）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# Step 3: ReContrast ResNet版 对比
python recontrast_wafer.py --dataset wafer

# Step 4: 🔥 ReContrast ViT+DINOv2 对比（新增）
python recontrast_vit_wafer.py --dataset wafer

# Step 5: DenseSimSiam 对比
python train_simsiam.py --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
```

### 场景B：MVTec AD 标准验证

```bash
# ViT+MoCo 全15类
python -m wafer_defect_detection.train --mode train_eval_all --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# ReContrast ResNet版 全品类
python recontrast_wafer.py --dataset mvtec

# 🔥 ReContrast ViT+DINOv2 全品类（新增）
python recontrast_vit_wafer.py --dataset mvtec
```

### 场景C：快速验证（10-20 epoch）

```bash
# ViT+MoCo 快速测试（指定一个品类的一个视图，每5轮评估）
python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 10 --batch_size 16 --wafer_category "BGA S5E 16x7" --wafer_view UP --eval_interval 5 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere

# 消融快速测试
python run_ablation.py --data_dir ./data --save_dir ./ablation_results_quick --epochs 10 --batch_size 16

# ReContrast ResNet版 快速测试（1000 iters）
python recontrast_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP

# 🔥 ReContrast ViT+DINOv2 快速测试（新增）
python recontrast_vit_wafer.py --dataset wafer --wafer_category "BGA S5E 16x7" --wafer_view UP
```

---

## ⚙️ 七、参数速查（原六）

### ViT+MoCo V3 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | train | train / eval / train_eval_all / train_all_wafer |
| `--dataset` | wafer | wafer / mvtec |
| `--data_dir` | ./data | 晶圆数据集根目录 |
| `--wafer_category` | None | 晶圆品类名，如 "BGA S5E 16x7" |
| `--wafer_view` | ALL | ALL / UP / DOWN |
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
| `--lr` | 1e-3 | 学习率（AdamW） |
| `--num_workers` | 4 | 数据加载线程 |
| `--seed` | 42 | 随机种子 |
| `--val_ratio` | 0.2 | 验证集比例 |
| `--eval_interval` | **10** 🔥 | 自动评估间隔（epoch），每10轮打印AUROC/F1/混淆矩阵 |
| `--save_dir` | ./checkpoints_v3_baseline/mycode | 模型保存目录（品类训练时自动创建子文件夹） |
| `--checkpoint` | — | 评估时用的checkpoint路径（自动在子文件夹中查找） |
| `--pca_components` | None | PCA降维维度 |
| `--score_mode` | combined | combined / mahal / memory / max |

### ReContrast（ResNet版）参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec | mvtec / wafer |
| `--categories` | grid,tile,screw | 品类列表，逗号分隔 |
| `--wafer_category` | None | 单个晶圆品类 |
| `--wafer_view` | ALL | ALL / UP / DOWN |
| `--wafer_data_dir` | ./data | 晶圆数据根目录 |
| `--save_dir` | ./checkpoints_v3_baseline/recontrast | 模型和结果保存目录（品类训练时自动创建子文件夹） |
| `--save_name` | recontrast_wafer | 实验命名（日志用） |
| `--gpu` | 0 | GPU ID |

### 🔥 ReContrast ViT + DINOv2 参数（新增）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | mvtec | mvtec / wafer |
| `--categories` | None | 品类列表，逗号分隔 |
| `--wafer_category` | None | 单个晶圆品类 |
| `--wafer_view` | ALL | ALL / UP / DOWN |
| `--wafer_data_dir` | ./data | 晶圆数据根目录 |
| `--save_dir` | ./checkpoints_vit_recontrast | 模型和结果保存目录 |
| `--save_name` | recontrast_vit | 实验命名（日志用） |
| `--use_wafer_encoder` | False | 使用现有ViTEncoder（而非DINOv2） |
| `--pretrained_model` | vit_small_patch14_dinov2.lvd142m | DINOv2预训练模型名 |
| `--eval_interval` | **100** 🔥 | 评估间隔（iters），默认100 |
| `--gpu` | 0 | GPU ID |

**预训练模型选项：**
- `vit_small_patch14_dinov2.lvd142m` (22.1M, 384dim) ⭐⭐⭐⭐⭐ 推荐
- `vit_small_patch16_224.dino` (21.7M, 384dim) ⭐⭐⭐⭐
- `vit_small_patch16_224.augreg_in21k` (21.7M, 384dim) ⭐⭐⭐

### 消融实验参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_range` | None | 指定实验范围，如 "0-2" 或 "3,5" |
| `--save_dir` | ./checkpoints_v3_baseline/ablation | 结果保存目录（自动创建Exp子文件夹） |
| 其他 | 同 ViT+MoCo | 继承主模型所有参数 |

### DenseSimSiam 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | train | train / eval / all / train_eval_all |
| `--use_dense` | True | 稠密(patch级)SimSiam |
| `--proj_dim` | 256 | SimSiam投影维度 |
| `--pred_hidden` | 128 | SimSiam预测器隐藏维度 |

---

> 🟡🦖 奶龙整理完毕！新增 ReContrast ViT + DINOv2 预训练权重支持！
> 现在可以跑三种对比方法：ResNet版 / ViT+DINOv2版 / DenseSimSiam
> 所有命令都是直接复制粘贴就能跑哒～ 好朋友加油写论文嗷呜！✨
