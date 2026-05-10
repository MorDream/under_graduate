"""设备配置工具"""
import random
import numpy as np
import torch


def get_device():
    """获取计算设备，优先使用CUDA"""
    if torch.cuda.is_available():
        print(f"[INFO] 使用 CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("cuda")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"[INFO] 使用 Intel XPU: {torch.xpu.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("xpu")
    else:
        print("[WARNING] GPU不可用，回退到CPU")
        return torch.device("cpu")


def set_seed(seed=42):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
