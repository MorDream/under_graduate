@echo off
chcp 65001 >nul
echo ========================================
echo  ViT+MoCo 改进版 V3 - 快速启动脚本
echo ========================================
echo.

:: 设置路径
set CODE_DIR=C:\Users\21196\Desktop\毕业设计\code
set PYTHON=python

cd /d %CODE_DIR%

echo 选择操作:
echo.
echo [1] 下载MVTec AD数据集
echo [2] 训练晶圆数据集
echo [3] 训练MVTec AD (单类别)
echo [4] 训练MVTec AD (全部类别)
echo [5] 评估模型
echo [6] 退出
echo.

set /p choice=请输入选项 (1-6): 

if "%choice%"=="1" goto download
if "%choice%"=="2" goto train_wafer
if "%choice%"=="3" goto train_mvtec_single
if "%choice%"=="4" goto train_mvtec_all
if "%choice%"=="5" goto evaluate
if "%choice%"=="6" goto end

echo 无效选项!
pause
goto end

:download
echo.
echo 正在下载MVTec AD数据集...
%PYTHON% download_mvtec.py
pause
goto end

:train_wafer
echo.
echo 训练晶圆数据集...
%PYTHON% train_improved_v3.py ^
    --dataset wafer ^
    --data_dir ./data ^
    --epochs 100 ^
    --batch_size 32 ^
    --use_cutpaste ^
    --use_feature_generator ^
    --use_hypersphere
pause
goto end

:train_mvtec_single
echo.
echo 可用的MVTec类别:
echo   bottle, cable, capsule, carpet, grid
echo   hazelnut, leather, metal_nut, pill, screw
echo   tile, toothbrush, transistor, wood, zipper
echo.
set /p category=请输入类别名 (默认: bottle): 
if "%category%"=="" set category=bottle

echo.
echo 正在训练 MVTec AD - %category%...
%PYTHON% train_improved_v3.py ^
    --dataset mvtec ^
    --mvtec_dir ./mvtec_anomaly_detection ^
    --mvtec_category %category% ^
    --epochs 100 ^
    --batch_size 32 ^
    --use_cutpaste ^
    --use_feature_generator ^
    --use_hypersphere
pause
goto end

:train_mvtec_all
echo.
echo 警告: 这将训练所有15个类别，可能需要2-4小时!
echo 按Ctrl+C取消，或按任意键继续...
pause >nul

echo.
echo 正在训练 MVTec AD 所有类别...
%PYTHON% train_improved_v3.py ^
    --mode train_eval_all ^
    --dataset mvtec ^
    --mvtec_dir ./mvtec_anomaly_detection ^
    --epochs 100 ^
    --batch_size 32
pause
goto end

:evaluate
echo.
set /p dataset=选择数据集 (wafer/mvtec): 

if "%dataset%"=="wafer" (
    %PYTHON% train_improved_v3.py ^
        --mode eval ^
        --dataset wafer ^
        --checkpoint ./checkpoints_v3/best_model_wafer.pth
) else if "%dataset%"=="mvtec" (
    set /p category=请输入类别名: 
    %PYTHON% train_improved_v3.py ^
        --mode eval ^
        --dataset mvtec ^
        --mvtec_category %category% ^
        --checkpoint ./checkpoints_v3/mvtec_%category%/best_model_mvtec.pth
) else (
    echo 无效的数据集!
)
pause
goto end

:end
echo.
echo 奶龙说: 嗷呜！完成任务啦！🟡🦖
