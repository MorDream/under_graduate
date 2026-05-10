#!/bin/bash
# ViT+MoCo 改进版 V3 - Linux/Mac 快速启动脚本

echo "========================================"
echo " ViT+MoCo 改进版 V3 - 快速启动脚本"
echo "========================================"
echo ""

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CODE_DIR"

PYTHON=${PYTHON:-python3}

echo "选择操作:"
echo ""
echo "[1] 下载MVTec AD数据集"
echo "[2] 训练晶圆数据集"
echo "[3] 训练MVTec AD (单类别)"
echo "[4] 训练MVTec AD (全部类别)"
echo "[5] 评估模型"
echo "[6] 退出"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "正在下载MVTec AD数据集..."
        $PYTHON download_mvtec.py
        ;;
    2)
        echo ""
        echo "训练晶圆数据集..."
        $PYTHON train_improved_v3.py \
            --dataset wafer \
            --data_dir ./data \
            --epochs 100 \
            --batch_size 32 \
            --use_cutpaste \
            --use_feature_generator \
            --use_hypersphere
        ;;
    3)
        echo ""
        echo "可用的MVTec类别:"
        echo "  bottle, cable, capsule, carpet, grid"
        echo "  hazelnut, leather, metal_nut, pill, screw"
        echo "  tile, toothbrush, transistor, wood, zipper"
        echo ""
        read -p "请输入类别名 (默认: bottle): " category
        category=${category:-bottle}
        
        echo ""
        echo "正在训练 MVTec AD - $category..."
        $PYTHON train_improved_v3.py \
            --dataset mvtec \
            --mvtec_dir ./mvtec_anomaly_detection \
            --mvtec_category $category \
            --epochs 100 \
            --batch_size 32 \
            --use_cutpaste \
            --use_feature_generator \
            --use_hypersphere
        ;;
    4)
        echo ""
        echo "警告: 这将训练所有15个类别，可能需要2-4小时!"
        read -p "按Enter键继续，或按Ctrl+C取消..."
        
        echo ""
        echo "正在训练 MVTec AD 所有类别..."
        $PYTHON train_improved_v3.py \
            --mode train_eval_all \
            --dataset mvtec \
            --mvtec_dir ./mvtec_anomaly_detection \
            --epochs 100 \
            --batch_size 32
        ;;
    5)
        echo ""
        read -p "选择数据集 (wafer/mvtec): " dataset
        
        if [ "$dataset" == "wafer" ]; then
            $PYTHON train_improved_v3.py \
                --mode eval \
                --dataset wafer \
                --checkpoint ./checkpoints_v3/best_model_wafer.pth
        elif [ "$dataset" == "mvtec" ]; then
            read -p "请输入类别名: " category
            $PYTHON train_improved_v3.py \
                --mode eval \
                --dataset mvtec \
                --mvtec_category $category \
                --checkpoint ./checkpoints_v3/mvtec_$category/best_model_mvtec.pth
        else
            echo "无效的数据集!"
        fi
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项!"
        ;;
esac

echo ""
echo "奶龙说: 嗷呜！完成任务啦！🟡🦖"
