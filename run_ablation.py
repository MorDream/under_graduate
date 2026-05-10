"""
消融实验脚本 - 验证各种改进的效果
比较不同配置在MVTec AD上的性能
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
import sys


def run_experiment(config_name, args_dict, category='bottle', epochs=50):
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"实验: {config_name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        sys.executable, 'train_improved_v3.py',
        '--mode', 'train',
        '--dataset', 'mvtec',
        '--mvtec_dir', './mvtec_anomaly_detection',
        '--mvtec_category', category,
        '--epochs', str(epochs),
        '--batch_size', '32',
        '--save_dir', f'./checkpoints_v3/ablation_{config_name}',
    ]
    
    # 添加额外参数
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
            else:
                cmd.append(f'--no-{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # 运行训练
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ 训练失败: {result.stderr}")
        return None
    
    # 评估
    eval_cmd = [
        sys.executable, 'train_improved_v3.py',
        '--mode', 'eval',
        '--dataset', 'mvtec',
        '--mvtec_category', category,
        '--checkpoint', f'./checkpoints_v3/ablation_{config_name}/best_model_mvtec.pth',
        '--save_dir', f'./checkpoints_v3/ablation_{config_name}',
    ]
    
    print(f"评估命令: {' '.join(eval_cmd)}")
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    if eval_result.returncode != 0:
        print(f"✗ 评估失败: {eval_result.stderr}")
        return None
    
    # 读取结果
    result_file = Path(f'./checkpoints_v3/ablation_{config_name}/results_mvtec_{category}.json')
    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)
        print(f"✓ 实验完成: AUROC={results['auroc']:.4f}, F1={results['f1']:.4f}")
        return results
    else:
        print(f"✗ 未找到结果文件")
        return None


def main():
    """运行消融实验"""
    
    # 实验配置
    experiments = [
        {
            'name': 'baseline',
            'desc': 'MoCo基线 (无改进)',
            'args': {
                'use_cutpaste': False,
                'use_feature_generator': False,
                'use_hypersphere': False,
            }
        },
        {
            'name': 'cutpaste',
            'desc': 'MoCo + CutPaste',
            'args': {
                'use_cutpaste': True,
                'use_feature_generator': False,
                'use_hypersphere': False,
            }
        },
        {
            'name': 'hypersphere',
            'desc': 'MoCo + 超球面约束',
            'args': {
                'use_cutpaste': False,
                'use_feature_generator': False,
                'use_hypersphere': True,
            }
        },
        {
            'name': 'generator',
            'desc': 'MoCo + 特征生成器',
            'args': {
                'use_cutpaste': False,
                'use_feature_generator': True,
                'use_hypersphere': False,
            }
        },
        {
            'name': 'cutpaste_hypersphere',
            'desc': 'MoCo + CutPaste + 超球面',
            'args': {
                'use_cutpaste': True,
                'use_feature_generator': False,
                'use_hypersphere': True,
            }
        },
        {
            'name': 'full',
            'desc': '完整版本 (所有改进)',
            'args': {
                'use_cutpaste': True,
                'use_feature_generator': True,
                'use_hypersphere': True,
            }
        },
    ]
    
    # 运行所有实验
    category = 'bottle'  # 使用bottle类别进行快速验证
    epochs = 50  # 减少轮数以加快验证
    
    results = []
    
    print("="*60)
    print("消融实验 - 验证改进效果")
    print("="*60)
    print(f"测试类别: {category}")
    print(f"训练轮数: {epochs}")
    print(f"总实验数: {len(experiments)}")
    print("="*60)
    
    for exp in experiments:
        result = run_experiment(exp['name'], exp['args'], category, epochs)
        if result:
            results.append({
                'name': exp['name'],
                'desc': exp['desc'],
                'auroc': result['auroc'],
                'f1': result['f1'],
                'accuracy': result['accuracy'],
            })
    
    # 汇总结果
    if results:
        print("\n" + "="*60)
        print("消融实验结果汇总")
        print("="*60)
        print(f"{'配置':<30} {'AUROC':<10} {'F1':<10} {'Accuracy':<10}")
        print("-"*60)
        
        for r in results:
            print(f"{r['desc']:<30} {r['auroc']:<10.4f} {r['f1']:<10.4f} {r['accuracy']:<10.4f}")
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv('./ablation_results.csv', index=False)
        print("\n结果已保存到: ./ablation_results.csv")
        
        # 绘制对比图
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            names = [r['name'] for r in results]
            aurocs = [r['auroc'] for r in results]
            f1s = [r['f1'] for r in results]
            
            axes[0].bar(names, aurocs, color='steelblue')
            axes[0].set_ylabel('AUROC')
            axes[0].set_title('AUROC Comparison')
            axes[0].set_ylim([0.5, 1.0])
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(names, f1s, color='coral')
            axes[1].set_ylabel('F1 Score')
            axes[1].set_title('F1 Score Comparison')
            axes[1].set_ylim([0.5, 1.0])
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('./ablation_comparison.png', dpi=150, bbox_inches='tight')
            print("对比图已保存到: ./ablation_comparison.png")
        except Exception as e:
            print(f"绘图失败: {e}")
    else:
        print("\n没有收集到有效结果")


if __name__ == '__main__':
    main()
