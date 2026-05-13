"""
train_improved_v3.py — 桥接文件
将 run.bat / run.sh 的命令转发到模块化版本
等价于: python -m wafer_defect_detection.train [args...]
"""
import sys
from pathlib import Path

# 确保 code 目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from wafer_defect_detection.train import main

if __name__ == '__main__':
    main()
