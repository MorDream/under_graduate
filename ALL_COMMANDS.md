     1|# 🟡 毕业设计 - 所有数据集 / 消融实验命令清单
     2|
     3|> 工作目录：`C:\Users\21196\Desktop\毕业设计\code`（即本文件所在目录）
     4|> 所有命令在命令行中从 `code` 目录执行
     5|
     6|---
     7|
     8|## 📂 目录结构速览
     9|
    10|```
    11|code/
    12|├── wafer_defect_detection/train.py    ← ViT+MoCo V3 主模型（模块化版本）
    13|├── train_simsiam.py                   ← DenseSimSiam 对比方案
    14|├── run_ablation.py                    ← 消融实验（6个实验逐步添加模块）
    15|├── baseline/train.py                  ← 早期 Baseline 版本
    16|├── download_mvtec.py                  ← 下载 MVTec AD 数据集
    17|├── data/数据集/数据集/                 ← 晶圆数据集（10种产品类型）
    18|├── mvtec_anomaly_detection/           ← MVTec AD 数据集
    19|├── checkpoints_v3/                    ← ViT+MoCo 模型保存
    20|├── checkpoints_simsiam/               ← DenseSimSiam 模型保存
    21|└── ablation_results/                  ← 消融实验输出
    22|```
    23|
    24|---
    25|
    26|## 🏭 可用数据集
    27|
    28|### 晶圆数据集（半导体产品质量检测）
    29|| 产品类型 | 目录名 |
    30||---------|--------|
    31|| BGA 12x4 | data/数据集/数据集/BGA 12x4 |
    32|| BGA S5E 16x7 | data/数据集/数据集/BGA S5E 16x7 |
    33|| ESSD 12x4 | data/数据集/数据集/ESSD 12x4 |
    34|| ESSD 12x5 | data/数据集/数据集/ESSD 12x5 |
    35|| INAND 16x5 | data/数据集/数据集/INAND 16x5 |
    36|| INAND 19x5 | data/数据集/数据集/INAND 19x5 |
    37|| MicroSD 20x4 | data/数据集/数据集/MicroSD 20x4 |
    38|| SDSIP 22x3 | data/数据集/数据集/SDSIP 22x3 |
    39|| UBGA 12x5 | data/数据集/数据集/UBGA 12x5 |
    40|| Defect sample | data/数据集/数据集/Defect sample |
    41|
    42|### MVTec AD 标准数据集（15个类别）
    43|```
    44|bottle, cable, capsule, carpet, grid,
    45|hazelnut, leather, metal_nut, pill, screw,
    46|tile, toothbrush, transistor, wood, zipper
    47|```
    48|
    49|---
    50|
    51|## 🔬 一、ViT + MoCo V3（主模型）
    52|
    53|> 入口脚本：`python -m wafer_defect_detection.train`（模块化版本）
    54|> ⚠️ run.bat/run.sh 中引用的 `train_improved_v3.py` 需改为 `-m wafer_defect_detection.train`
    55|
    56|### 1.1 晶圆数据集 - 训练
    57|
    58|```bash
    59|# 基础训练（200 epoch，启用全部改进模块）
    60|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    61|
    62|# 快速测试（20 epoch，小batch）
    63|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 20 --batch_size 16 --lr 1e-3 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    64|
    65|# 基础版（关闭所有改进，纯ViT+MoCo）
    66|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3
    67|
    68|# 不划分验证集（全部数据用于训练）
    69|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 200 --batch_size 32 --lr 1e-3 --val_ratio 0.0 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    70|
    71|# 自定义超参数
    72|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --epochs 300 --batch_size 64 --lr 5e-4 --embed_dim 512 --queue_size 2048 --temperature 0.1 --hypersphere_weight 0.2 --discriminator_weight 0.1 --generator_weight 0.1 --cutpaste_prob 0.5 --seed 123 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    73|
    74|# 不同图像尺寸
    75|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 128 --epochs 200 --batch_size 64 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    76|
    77|python -m wafer_defect_detection.train --mode train --dataset wafer --data_dir ./data --img_size 384 --epochs 200 --batch_size 16 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
    78|```
    79|
    80|### 1.2 晶圆数据集 - 评估
    81|
    82|```bash
    83|# 使用自动找到的最佳模型评估
    84|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --save_dir ./checkpoints_v3
    85|
    86|# 指定checkpoint路径评估
    87|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_v3/best_model_wafer.pth
    88|
    89|# 不同评分模式评估
    90|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode combined --checkpoint ./checkpoints_v3/best_model_wafer.pth
    91|
    92|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode mahal --checkpoint ./checkpoints_v3/best_model_wafer.pth
    93|
    94|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode memory --checkpoint ./checkpoints_v3/best_model_wafer.pth
    95|
    96|python -m wafer_defect_detection.train --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --score_mode max --checkpoint ./checkpoints_v3/best_model_wafer.pth
    97|```
    98|
    99|### 1.3 MVTec AD - 单类别训练+评估
   100|
   101|```bash
   102|# 训练 bottle（默认）
   103|python -m wafer_defect_detection.train  104|    --mode train  105|    --dataset mvtec  106|    --mvtec_dir ./mvtec_anomaly_detection  107|    --mvtec_category bottle  108|    --epochs 200  109|    --batch_size 32  110|    --use_cutpaste  111|    --use_multiscale  112|    --use_feature_generator  113|    --use_hypersphere
   114|
   115|# 训练 cable
   116|python -m wafer_defect_detection.train  117|    --mode train  118|    --dataset mvtec  119|    --mvtec_dir ./mvtec_anomaly_detection  120|    --mvtec_category cable  121|    --epochs 200  122|    --batch_size 32  123|    --use_cutpaste  124|    --use_multiscale  125|    --use_feature_generator  126|    --use_hypersphere
   127|
   128|# === MVTec 15 个类别全部训练命令（粘贴即用）===
   129|
   130|# bottle
   131|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   132|
   133|# cable
   134|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   135|
   136|# capsule
   137|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   138|
   139|# carpet
   140|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   141|
   142|# grid
   143|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   144|
   145|# hazelnut
   146|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   147|
   148|# leather
   149|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   150|
   151|# metal_nut
   152|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   153|
   154|# pill
   155|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   156|
   157|# screw
   158|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   159|
   160|# tile
   161|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   162|
   163|# toothbrush
   164|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   165|
   166|# transistor
   167|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   168|
   169|# wood
   170|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   171|
   172|# zipper
   173|python -m wafer_defect_detection.train --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_feature_generator --use_hypersphere
   174|```
   175|
   176|### 1.4 MVTec AD - 单类别评估
   177|
   178|```bash
   179|# 评估 bottle
   180|python -m wafer_defect_detection.train  181|    --mode eval  182|    --dataset mvtec  183|    --mvtec_dir ./mvtec_anomaly_detection  184|    --mvtec_category bottle  185|    --checkpoint ./checkpoints_v3/best_model_mvtec.pth  186|    --score_mode combined
   187|
   188|# === MVTec 15 类评估命令 ===
   189|# bottle
   190|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   191|
   192|# cable
   193|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   194|
   195|# capsule
   196|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   197|
   198|# carpet
   199|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   200|
   201|# grid
   202|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   203|
   204|# hazelnut
   205|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   206|
   207|# leather
   208|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   209|
   210|# metal_nut
   211|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   212|
   213|# pill
   214|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   215|
   216|# screw
   217|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   218|
   219|# tile
   220|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   221|
   222|# toothbrush
   223|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   224|
   225|# transistor
   226|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   227|
   228|# wood
   229|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   230|
   231|# zipper
   232|python -m wafer_defect_detection.train --mode eval --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --checkpoint ./checkpoints_v3/best_model_mvtec.pth --score_mode combined
   233|```
   234|
   235|### 1.5 MVTec AD - 一键训练+评估所有15类
   236|
   237|```bash
   238|# 自动遍历所有15个类别，训练→评估→汇总
   239|python -m wafer_defect_detection.train  240|    --mode train_eval_all  241|    --dataset mvtec  242|    --mvtec_dir ./mvtec_anomaly_detection  243|    --epochs 200  244|    --batch_size 32  245|    --use_cutpaste  246|    --use_multiscale  247|    --use_feature_generator  248|    --use_hypersphere
   249|
   250|# 快速版（50epoch）
   251|python -m wafer_defect_detection.train  252|    --mode train_eval_all  253|    --dataset mvtec  254|    --mvtec_dir ./mvtec_anomaly_detection  255|    --epochs 50  256|    --batch_size 32  257|    --use_cutpaste  258|    --use_multiscale  259|    --use_feature_generator  260|    --use_hypersphere
   261|```
   262|
   263|---
   264|
   265|## 🔬 二、DenseSimSiam（对比方案）
   266|
   267|> 入口脚本：`python train_simsiam.py`
   268|> SimSiam 无动量编码器+无负样本队列，节省50%显存
   269|
   270|### 2.1 晶圆数据集 - 训练
   271|
   272|```bash
   273|# 完整训练
   274|python train_simsiam.py  275|    --mode train  276|    --dataset wafer  277|    --data_dir ./data  278|    --epochs 200  279|    --batch_size 32  280|    --lr 1e-3  281|    --use_cutpaste  282|    --use_multiscale  283|    --use_dense  284|    --use_feature_generator  285|    --use_hypersphere
   286|
   287|# 基础版（关闭辅助模块）
   288|python train_simsiam.py  289|    --mode train  290|    --dataset wafer  291|    --data_dir ./data  292|    --epochs 200  293|    --batch_size 32
   294|
   295|# 仅全局 SimSiam（无多尺度、无稠密）
   296|python train_simsiam.py  297|    --mode train  298|    --dataset wafer  299|    --data_dir ./data  300|    --epochs 200  301|    --batch_size 32  302|    --use_multiscale  303|    --use_dense
   304|
   305|# 仅稠密 SimSiam（无多尺度）
   306|python train_simsiam.py  307|    --mode train  308|    --dataset wafer  309|    --data_dir ./data  310|    --epochs 200  311|    --batch_size 32  312|    --use_dense
   313|
   314|# 仅多尺度 SimSiam（无稠密）
   315|python train_simsiam.py  316|    --mode train  317|    --dataset wafer  318|    --data_dir ./data  319|    --epochs 200  320|    --batch_size 32  321|    --use_multiscale
   322|
   323|# 自定义损失权重
   324|python train_simsiam.py  325|    --mode train  326|    --dataset wafer  327|    --data_dir ./data  328|    --epochs 200  329|    --batch_size 32  330|    --global_weight 1.0  331|    --dense_weight 0.5  332|    --multiscale_weight 0.5  333|    --hypersphere_weight 0.2  334|    --discriminator_weight 0.1  335|    --generator_weight 0.1  336|    --use_cutpaste  337|    --use_multiscale  338|    --use_dense  339|    --use_feature_generator  340|    --use_hypersphere
   341|
   342|# 自定义模型维度
   343|python train_simsiam.py  344|    --mode train  345|    --dataset wafer  346|    --data_dir ./data  347|    --epochs 200  348|    --batch_size 32  349|    --embed_dim 512  350|    --proj_dim 256  351|    --pred_hidden 128  352|    --use_cutpaste  353|    --use_multiscale  354|    --use_dense  355|    --use_feature_generator  356|    --use_hypersphere
   357|
   358|# 不同图像尺寸
   359|python train_simsiam.py  360|    --mode train  361|    --dataset wafer  362|    --data_dir ./data  363|    --img_size 128  364|    --epochs 200  365|    --batch_size 64  366|    --use_cutpaste  367|    --use_multiscale  368|    --use_dense  369|    --use_feature_generator  370|    --use_hypersphere
   371|```
   372|
   373|### 2.2 晶圆数据集 - 评估
   374|
   375|```bash
   376|# 自动找checkpoint
   377|python train_simsiam.py  378|    --mode eval  379|    --dataset wafer  380|    --data_dir ./data  381|    --val_ratio 0.2  382|    --save_dir ./checkpoints_simsiam  383|    --score_mode combined
   384|
   385|# 指定checkpoint
   386|python train_simsiam.py  387|    --mode eval  388|    --dataset wafer  389|    --data_dir ./data  390|    --val_ratio 0.2  391|    --checkpoint ./checkpoints_simsiam/best_model_wafer.pth  392|    --score_mode combined
   393|
   394|# 不同评分模式
   395|python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode mahal
   396|python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode memory
   397|python train_simsiam.py --mode eval --dataset wafer --data_dir ./data --val_ratio 0.2 --checkpoint ./checkpoints_simsiam/best_model_wafer.pth --score_mode max
   398|```
   399|
   400|### 2.3 晶圆数据集 - 训练+评估一条龙
   401|
   402|```bash
   403|python train_simsiam.py  404|    --mode all  405|    --dataset wafer  406|    --data_dir ./data  407|    --epochs 200  408|    --batch_size 32  409|    --use_cutpaste  410|    --use_multiscale  411|    --use_dense  412|    --use_feature_generator  413|    --use_hypersphere
   414|```
   415|
   416|### 2.4 MVTec AD - 单类别训练
   417|
   418|```bash
   419|# === DenseSimSiam MVTec 15类训练 ===
   420|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category bottle --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   421|
   422|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category cable --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   423|
   424|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category capsule --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   425|
   426|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category carpet --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   427|
   428|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category grid --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   429|
   430|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category hazelnut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   431|
   432|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category leather --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   433|
   434|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category metal_nut --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   435|
   436|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category pill --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   437|
   438|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category screw --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   439|
   440|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category tile --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   441|
   442|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category toothbrush --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   443|
   444|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category transistor --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   445|
   446|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category wood --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   447|
   448|python train_simsiam.py --mode train --dataset mvtec --mvtec_dir ./mvtec_anomaly_detection --mvtec_category zipper --epochs 200 --batch_size 32 --use_cutpaste --use_multiscale --use_dense --use_feature_generator --use_hypersphere
   449|```
   450|
   451|### 2.5 MVTec AD - 单类别评估
   452|
   453|```bash
   454|# bottle 评估
   455|python train_simsiam.py  456|    --mode eval  457|    --dataset mvtec  458|    --mvtec_dir ./mvtec_anomaly_detection  459|    --mvtec_category bottle  460|    --save_dir ./checkpoints_simsiam  461|    --score_mode combined
   462|```
   463|
   464|### 2.6 MVTec AD - 一键训练+评估所有15类
   465|
   466|```bash
   467|# DenseSimSiam 全类别
   468|python train_simsiam.py  469|    --mode train_eval_all  470|    --dataset mvtec  471|    --mvtec_dir ./mvtec_anomaly_detection  472|    --epochs 200  473|    --batch_size 32  474|    --use_cutpaste  475|    --use_multiscale  476|    --use_dense  477|    --use_feature_generator  478|    --use_hypersphere
   479|```
   480|
   481|---
   482|
   483|## 🧪 三、消融实验（run_ablation.py）
   484|
   485|> 入口脚本：`python run_ablation.py`
   486|> 6个实验逐步添加模块，验证各模块贡献
   487|
   488|### 实验设计
   489|| 实验 | 描述 | 增量模块 |
   490||------|------|---------|
   491|| Exp0 | ViT+MoCo（纯baseline） | — |
   492|| Exp1 | + 多尺度特征融合 | 多尺度 |
   493|| Exp2 | + CutPaste合成异常 | CutPaste |
   494|| Exp3 | + 超球面约束(CFA) | CFA |
   495|| Exp4 | + 特征生成-判别(SimpleNet) | SimpleNet |
   496|| Exp5 | + 记忆库(PatchCore) = 完整版 | 记忆库 |
   497|
   498|### 3.1 运行全部6个实验（默认）
   499|
   500|```bash
   501|