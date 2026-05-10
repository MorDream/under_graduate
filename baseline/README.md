# 半导体晶圆无监督缺陷检测 Baseline - 改进版

## 改进概览

基于开题报告分析的问题，本版本实现了以下改进：

### 改进点1：多尺度特征融合（解决特征淹没）
**问题**：全局对比损失将优化重点放在大尺度背景特征，微小缺陷特征被淹没。

**解决方案**：
- 新增 `forward_multiscale()` 方法，融合CLS token + 各层patch token特征
- 收集第2、4、6层（偶数层）的中间特征
- 通过平均池化融合多尺度特征

**代码位置**：
- `ViTEncoder.forward()` 增加 `return_all_layers` 参数
- `ViTEncoder.forward_multiscale()` 新方法
- `MoCoV2.forward()` 中调用多尺度特征提取

### 改进点2：优化数据增强策略（半导体专用）
**问题**：通用增强（颜色抖动）可能破坏晶圆图的物理语义。

**解决方案**：
- 移除破坏性颜色变换，保留原始像素物理语义
- 增加半导体特有增强：
  - 高斯模糊模拟聚焦不准
  - 局部遮挡模拟颗粒污染（RandomErasing with small scale）
  - 多尺度遮挡模拟复杂缺陷
- 新增 `SemiconductorStrongTransform` 强增强类

**代码位置**：
- `SemiconductorTransform` 类重构
- `SemiconductorStrongTransform` 新类

### 改进点3：改进异常检测机制（PCA + Ledoit-Wolf）
**问题**：马氏距离计算协方差矩阵不稳定，高维特征计算困难。

**解决方案**：
- **PCA降维**：自动选择保留95%方差的维度，减少噪声特征
- **Ledoit-Wolf收缩估计**：更稳健的协方差矩阵估计
  - 目标矩阵：对角方差矩阵
  - 收缩系数：基于样本数和特征数自适应计算
- 输出PCA维度和收缩系数信息

**代码位置**：
- `AnomalyDetector._ledoit_wolf_shrinkage()` 新方法
- `AnomalyDetector.fit()` 使用PCA降维
- `AnomalyDetector._pca_transform()` 辅助方法

### 改进点4：热力图定位改进（多层注意力聚合）
**问题**：仅使用最后一层注意力，定位精度有限。

**解决方案**：
- **多层注意力聚合**：加权融合所有层注意力，深层权重更高
- **GradCAM风格**：基于梯度计算热力图（保留接口）
- **异常分数图**：patch级别计算马氏距离，生成像素级定位

**代码位置**：
- `generate_attention_map()` 支持3种方法：
  - `'last'`：原方法（仅最后一层）
  - `'multilayer'`：多层加权聚合
  - `'gradcam'`：GradCAM风格
- `generate_anomaly_score_map()` 新方法：patch级异常分数

### 改进点5：预训练权重初始化
**问题**：随机初始化收敛慢，需要更多数据。

**解决方案**：
- 新增 `load_pretrained_weights()` 函数
- 支持加载外部预训练权重
- 自动匹配参数形状，过滤不兼容的层
- `--pretrained` 参数指定权重路径

**代码位置**：
- `load_pretrained_weights()` 函数
- `train()` 中加载预训练权重

### 改进点6：训练策略优化（温度调度）
**问题**：固定温度参数可能导致早期学习不稳定。

**解决方案**：
- **温度参数调度**：
  - 预热阶段（前25% epochs）：温度=0.1，学习更平滑
  - 训练后期：线性衰减到基础温度的70%
- 新增 `update_temperature()` 方法
- 训练日志显示当前温度值

**代码位置**：
- `MoCoV2.update_temperature()` 方法
- `MoCoV2.forward()` 中自动调整温度

---

## 技术栈

- **编码器**: Vision Transformer (ViT) - 轻量化版本 (embed_dim=384, depth=6)
- **对比学习**: MoCo v2 (Momentum Contrast) + 多尺度特征融合
- **异常检测**: PCA降维 + Ledoit-Wolf协方差估计 + 马氏距离
- **热力图**: 多层注意力聚合 + GradCAM风格 + 异常分数图
- **硬件**: Intel Arc B580 (XPU) / CPU回退

---

## 环境配置

```bash
# 创建虚拟环境
conda create -n graduate python=3.13 -y
conda activate graduate

# 安装PyTorch XPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 安装其他依赖
pip install scikit-learn matplotlib
```

---

## 使用方法

### 1. 训练对比学习模型（使用所有改进）

```powershell
# Windows PowerShell
cd "C:\Users\21196\Desktop\毕业设计\code"
conda activate graduate

# 使用多尺度特征融合 + 改进异常检测
python baseline/train.py --mode train --epochs 100 --use_multiscale

# 如果有预训练权重
python baseline/train.py --mode train --epochs 100 --pretrained path/to/weights.pth
```

### 2. 评估异常检测效果（生成热力图）

```powershell
# 评估并生成多层注意力热力图
python baseline/train.py --mode eval --generate_heatmap --heatmap_method multilayer

# 评估并生成所有类型热力图对比
python baseline/train.py --mode eval --generate_heatmap --heatmap_method all
```

### 3. 训练并评估

```powershell
python baseline/train.py --mode train_and_eval --epochs 100 --generate_heatmap
```

---

## 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据集根目录 |
| `--img_size` | 224 | 输入图片大小 |
| `--embed_dim` | 384 | ViT嵌入维度 |
| `--queue_size` | 1024 | MoCo负样本队列大小 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 0.03 | 学习率 |
| `--temperature` | 0.07 | 对比损失基础温度参数 |

### 改进参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_multiscale` | True | 使用多尺度特征融合（改进点1） |
| `--pca_components` | None | PCA降维维度，None为自动选择（改进点3） |
| `--pretrained` | None | 预训练权重路径（改进点5） |
| `--heatmap_method` | multilayer | 热力图方法：last/multilayer/gradcam/all（改进点4） |

---

## 项目结构

```
baseline/
├── train.py           # 主训练脚本（改进版）
├── README.md          # 说明文档
├── checkpoints/       # 模型保存目录
│   ├── best_model.pth
│   └── final_model.pth
└── heatmaps/          # 热力图输出目录
```

---

## 输出指标

- **AUROC**: 曲线下面积
- **F1 Score**: 最优阈值下的F1分数
- **漏检率(FNR)**: 缺陷样本被误判为正常的比例
- **误检率(FPR)**: 正常样本被误判为缺陷的比例

---

## 改进效果预期

| 改进点 | 预期效果 |
|--------|----------|
| 多尺度特征融合 | 提升微小缺陷检测能力 |
| 数据增强优化 | 更好的特征学习，避免捷径学习 |
| 异常检测改进 | 更稳定可靠的异常评分 |
| 热力图改进 | 更精确的缺陷定位 |
| 预训练权重 | 加速收敛，提升性能 |
| 温度调度 | 更稳定的训练过程 |
