"""
XPU 环境验证脚本
运行此脚本检查Intel GPU是否正确配置
"""

import sys

def check_xpu():
    print("=" * 60)
    print("Intel XPU 环境验证")
    print("=" * 60)
    
    # 1. 检查Python版本
    print(f"\n[1] Python版本: {sys.version}")
    
    # 2. 检查PyTorch
    try:
        import torch
        print(f"[2] PyTorch版本: {torch.__version__}")
    except ImportError:
        print("[2] PyTorch未安装!")
        return False
    
    # 3. 检查XPU支持
    if hasattr(torch, 'xpu'):
        print(f"[3] torch.xpu模块: 存在")
        
        if torch.xpu.is_available():
            print(f"[4] XPU可用: 是")
            
            # 获取设备信息
            device_count = torch.xpu.device_count()
            print(f"[5] XPU设备数量: {device_count}")
            
            if device_count > 0:
                device_name = torch.xpu.get_device_name(0)
                print(f"[6] 设备名称: {device_name}")
                
                props = torch.xpu.get_device_properties(0)
                print(f"[7] 显存大小: {props.total_memory / 1024**3:.2f} GB")
                
                # 简单测试
                print("\n[8] 运行简单测试...")
                try:
                    x = torch.randn(1000, 1000, device='xpu')
                    y = torch.randn(1000, 1000, device='xpu')
                    z = torch.mm(x, y)
                    torch.xpu.synchronize()
                    print("    矩阵乘法测试: 成功!")
                    print(f"    结果形状: {z.shape}")
                    return True
                except Exception as e:
                    print(f"    测试失败: {e}")
                    return False
        else:
            print("[4] XPU不可用")
            print("    可能原因: Intel GPU驱动未安装或设备不被支持")
            return False
    else:
        print("[3] torch.xpu模块: 不存在")
        print("    当前PyTorch版本可能不支持XPU")
        return False


if __name__ == "__main__":
    success = check_xpu()
    print("\n" + "=" * 60)
    if success:
        print("环境验证通过! 可以开始训练。")
    else:
        print("环境验证失败，请检查配置。")
    print("=" * 60)
