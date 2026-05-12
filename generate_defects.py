#!/usr/bin/env python3
"""
晶圆缺陷生成脚本 - 模仿真实半导体制造缺陷
==========================================
功能：从无缺陷样本生成合成缺陷，平衡评估集

缺陷类型（6种真实晶圆缺陷）:
1. 划痕 (Scratch) — 细长亮/暗线条
2. 颗粒污染 (Particle) — 小圆形亮/暗点
3. 污渍区域 (Stain) — 高斯模糊状局部亮暗变化
4. 多道划痕 (ScratchCluster) — 多道平行细线
5. 局部损坏 (LocalDamage) — 局部对比度/纹理突变
6. 边缘缺损 (EdgeDefect) — 图像边缘区域亮度异常

使用方法:
  python generate_defects.py
  python generate_defects.py --categories "BGA 12x4,BGA S5E 16x7"
  python generate_defects.py --dry-run   # 只看要生成多少
"""

import os
import sys
import random
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


# ============================================================
# 缺陷生成函数（每种返回一个新 PIL Image）
# ============================================================

def add_scratch(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加细划痕 — 随机位置、方向、宽度的亮/暗线条"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    canvas = np.zeros_like(arr)

    # 随机起点和终点
    margin = int(min(h, w) * 0.1)
    start_x = rng.randint(margin, w - margin)
    start_y = rng.randint(margin, h - margin)
    angle = rng.uniform(0, 2 * math.pi)
    length = rng.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.7))
    end_x = int(start_x + length * math.cos(angle))
    end_y = int(start_y + length * math.sin(angle))
    end_x = np.clip(end_x, 0, w - 1)
    end_y = np.clip(end_y, 0, h - 1)

    # 线条宽度
    width = rng.randint(1, 3)
    brightness = rng.choice([-1, 1]) * rng.uniform(40, 100)  # 亮或暗

    # 用 draw 画线
    img_pil = img.copy()
    draw = ImageDraw.Draw(img_pil)
    if brightness > 0:
        color = (rng.randint(180, 255),) * (len(arr.shape) if len(arr.shape) == 3 else 1)
    else:
        color = (rng.randint(0, 60),) * (len(arr.shape) if len(arr.shape) == 3 else 1)

    if len(arr.shape) == 3:
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
    else:
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color[0], width=width)

    # 轻微模糊让线条更自然
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img_pil


def add_particle(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加颗粒污染 — 小圆形亮/暗斑"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    img_pil = img.copy()
    draw = ImageDraw.Draw(img_pil)

    # 随机位置 1-3 个颗粒
    n_particles = rng.randint(1, 3)
    for _ in range(n_particles):
        cx = rng.randint(int(w * 0.15), int(w * 0.85))
        cy = rng.randint(int(h * 0.15), int(h * 0.85))
        radius = rng.randint(2, 8)
        brightness = rng.choice([-1, 1]) * rng.uniform(30, 80)

        if len(arr.shape) == 3:
            if brightness > 0:
                color = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255))
            else:
                color = (rng.randint(0, 40), rng.randint(0, 40), rng.randint(0, 40))
        else:
            if brightness > 0:
                color = rng.randint(200, 255)
            else:
                color = rng.randint(0, 40)

        # 画实心圆
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=color,
        )

    # 轻微模糊
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img_pil


def add_stain(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加污渍区域 — 高斯模糊的局部亮度变化区域"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # 生成随机高斯斑块
    stain_h = rng.randint(int(h * 0.1), int(h * 0.4))
    stain_w = rng.randint(int(w * 0.1), int(w * 0.4))
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))

    # 创建高斯斑块
    yy, xx = np.ogrid[:h, :w]
    sigma_x = stain_w / 3
    sigma_y = stain_h / 3
    gaussian = np.exp(-((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2)))

    # 斑块强度
    intensity = rng.uniform(-60, 60)  # 可亮可暗

    if len(arr.shape) == 3:
        for c in range(3):
            arr[:, :, c] += gaussian * intensity
    else:
        arr += gaussian * intensity

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_scratch_cluster(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加多道平行划痕"""
    img_pil = img.copy()
    draw = ImageDraw.Draw(img_pil)
    h, w = img.size  # PIL returns (w, h)
    # Note: img.size is (width, height)

    margin = int(min(h, w) * 0.15)
    base_x = rng.randint(margin, w - margin)
    base_y = rng.randint(margin, h - margin)
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    length = rng.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.5))
    n_lines = rng.randint(3, 6)
    spacing = rng.randint(3, 8)

    dx = length * math.cos(angle)
    dy = length * math.sin(angle)
    brightness = rng.choice([-1, 1])

    if len(np.array(img).shape) == 3:
        if brightness > 0:
            color = (rng.randint(200, 255),) * 3
        else:
            color = (rng.randint(0, 50),) * 3
    else:
        if brightness > 0:
            color = rng.randint(200, 255)
        else:
            color = rng.randint(0, 50)

    # 垂直于划线方向偏移
    perp_angle = angle + math.pi / 2
    for i in range(n_lines):
        offset = (i - n_lines // 2) * spacing
        ox = int(offset * math.cos(perp_angle))
        oy = int(offset * math.sin(perp_angle))
        x1 = np.clip(base_x + ox, 0, w - 1)
        y1 = np.clip(base_y + oy, 0, h - 1)
        x2 = np.clip(int(base_x + ox + dx), 0, w - 1)
        y2 = np.clip(int(base_y + oy + dy), 0, h - 1)

        if len(np.array(img).shape) == 3:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
        else:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    return img_pil.filter(ImageFilter.GaussianBlur(radius=0.3))


def add_local_damage(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加局部损坏 — 局部纹理/对比度突变"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # 随机矩形区域
    x1 = rng.randint(int(w * 0.1), int(w * 0.7))
    y1 = rng.randint(int(h * 0.1), int(h * 0.7))
    x2 = x1 + rng.randint(int(w * 0.05), int(w * 0.3))
    y2 = y1 + rng.randint(int(h * 0.05), int(h * 0.3))
    x2 = min(x2, w - 1)
    y2 = min(y2, h - 1)

    damage_type = rng.choice(['blur', 'brighten', 'darken', 'noise', 'contrast'])

    region = arr[y1:y2, x1:x2].copy()

    if damage_type == 'blur':
        # 高斯模糊
        kernel_size = rng.randint(3, 9)
        blurred = Image.fromarray(region.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=kernel_size / 2)
        )
        arr[y1:y2, x1:x2] = np.array(blurred).astype(np.float32)
    elif damage_type == 'brighten':
        arr[y1:y2, x1:x2] = np.clip(region + rng.uniform(30, 70), 0, 255)
    elif damage_type == 'darken':
        arr[y1:y2, x1:x2] = np.clip(region - rng.uniform(30, 70), 0, 255)
    elif damage_type == 'noise':
        noise = np.random.default_rng(rng.randint(0, 2**31)).uniform(-30, 30, region.shape)
        arr[y1:y2, x1:x2] = np.clip(region + noise, 0, 255)
    elif damage_type == 'contrast':
        # 提高对比度
        mean = region.mean()
        arr[y1:y2, x1:x2] = np.clip((region - mean) * rng.uniform(1.5, 3.0) + mean, 0, 255)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_edge_defect(img: Image.Image, rng: random.Random) -> Image.Image:
    """添加边缘缺损 — 图像边缘区域的亮度异常"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # 选择边缘方向
    edge = rng.choice(['top', 'bottom', 'left', 'right'])
    band_width = rng.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.2))
    intensity = rng.uniform(-80, 80)

    if edge == 'top':
        arr[:band_width, :] = np.clip(arr[:band_width, :] + intensity, 0, 255)
    elif edge == 'bottom':
        arr[-band_width:, :] = np.clip(arr[-band_width:, :] + intensity, 0, 255)
    elif edge == 'left':
        arr[:, :band_width] = np.clip(arr[:, :band_width] + intensity, 0, 255)
    elif edge == 'right':
        arr[:, -band_width:] = np.clip(arr[:, -band_width:] + intensity, 0, 255)

    # 渐变过渡使边缘自然
    mask = np.zeros_like(arr, dtype=np.float32)
    if edge in ['top', 'bottom']:
        for i in range(band_width):
            alpha = 1.0 - i / band_width
            if edge == 'bottom':
                idx = h - band_width + i
                arr[idx:idx+1, :] = arr[idx:idx+1, :] * alpha + arr[idx:idx+1, :] * (1 - alpha)
            else:
                arr[i:i+1, :] = arr[i:i+1, :] * alpha + arr[i:i+1, :] * (1 - alpha)
    else:
        for i in range(band_width):
            alpha = 1.0 - i / band_width
            if edge == 'right':
                idx = w - band_width + i
                arr[:, idx:idx+1] = arr[:, idx:idx+1] * alpha + arr[:, idx:idx+1] * (1 - alpha)
            else:
                arr[:, i:i+1] = arr[:, i:i+1] * alpha + arr[:, i:i+1] * (1 - alpha)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# 缺陷生成器列表
DEFECT_GENERATORS = [
    ("scratch", add_scratch),
    ("particle", add_particle),
    ("stain", add_stain),
    ("scratch_cluster", add_scratch_cluster),
    ("local_damage", add_local_damage),
    ("edge_defect", add_edge_defect),
]


# ============================================================
# 文件视图分类
# ============================================================

def get_view(filename: str) -> str:
    """根据文件名判断 UP/DOWN view"""
    name_upper = filename.upper()
    if '_UP' in name_upper:
        return 'UP'
    elif '_DOWN' in name_upper:
        return 'DOWN'
    return 'OTHER'


def collect_samples(data_root: str, category: str, subpath: str):
    """收集指定品类下指定子路径的所有图片"""
    full_path = os.path.join(data_root, category, subpath)
    if not os.path.isdir(full_path):
        return {}
    # 按 view 分组
    by_view = {'UP': [], 'DOWN': []}
    for f in sorted(os.listdir(full_path)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            view = get_view(f)
            if view in by_view:
                by_view[view].append(f)
    return by_view


# ============================================================
# 主逻辑
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='生成晶圆合成缺陷，平衡评估集')
    parser.add_argument('--data-root', type=str,
                        default='/mnt/c/Users/21196/Desktop/毕业设计/code/data/晶圆分类数据集',
                        help='晶圆分类数据集根目录')
    parser.add_argument('--categories', type=str, default=None,
                        help='要处理的品类（逗号分隔），默认全部')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dry-run', action='store_true',
                        help='只预览需要生成的数量，不实际生成')
    parser.add_argument('--min-defects', type=int, default=5,
                        help='每个(view, category)的最小缺陷数')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"[ERROR] 数据目录不存在: {data_root}")
        sys.exit(1)

    # 获取品类列表
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    else:
        categories = sorted([
            d.name for d in data_root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

    print(f"{'='*70}")
    print(f"晶圆合成缺陷生成")
    print(f"{'='*70}")
    print(f"数据根目录: {data_root}")
    print(f"品类数: {len(categories)}")
    if args.dry_run:
        print(f"[DRY RUN] 仅预览，不实际生成")
    print(f"{'='*70}\n")

    total_generated = 0
    total_by_type = defaultdict(int)

    for cat in categories:
        cat_path = data_root / cat
        if not cat_path.is_dir():
            continue

        train_good = collect_samples(str(data_root), cat, 'train/good')
        test_good = collect_samples(str(data_root), cat, 'test/good')
        test_defect = collect_samples(str(data_root), cat, 'test/defect')

        print(f"\n--- {cat} ---")

        for view in ['UP', 'DOWN']:
            n_train = len(train_good.get(view, []))
            n_good = len(test_good.get(view, []))
            n_def = len(test_defect.get(view, []))

            print(f"  [{view}] train/good={n_train:3d}  test/good={n_good:3d}  test/defect={n_def:3d}")

            if n_good == 0 and n_def >= args.min_defects:
                print(f"    → 无 test/good，缺陷{n_def}个够用，跳过")
                continue

            # 目标：test/defect ≈ max(test/good, min_defects)
            target = max(n_good, args.min_defects)
            needed = target - n_def

            if needed <= 0:
                print(f"    → 缺陷数已足够 (目标{target}, 已有{n_def})")
                continue

            # 从 train/good 取样本生成缺陷
            source_files = train_good.get(view, [])
            if len(source_files) == 0:
                print(f"    ⚠️  无训练正常样本可用，跳过")
                continue

            # 每个源文件最多生成几个缺陷
            # 需要 needed 个缺陷，有 n_train 个源文件
            n_per_source = max(1, math.ceil(needed / len(source_files)))
            to_generate = min(needed, len(source_files) * n_per_source)
            # 确保不超过 source files * reasonable multiplier
            max_per_source = min(n_per_source, 5)  # 每个源文件最多5个变体

            print(f"    → 需要 {needed} 个缺陷，将生成 {to_generate} 个")

            if args.dry_run:
                total_generated += to_generate
                continue

            # 实际生成
            defect_dir = cat_path / 'test' / 'defect'
            defect_dir.mkdir(parents=True, exist_ok=True)

            generated_count = 0
            attempts = 0

            # 打乱源文件顺序
            shuffled_sources = sorted(source_files)
            rng.shuffle(shuffled_sources)

            while generated_count < needed and attempts < needed * 3:
                for sf in shuffled_sources:
                    if generated_count >= needed:
                        break

                    src_path = cat_path / 'train' / 'good' / sf

                    try:
                        img = Image.open(src_path).convert('RGB')
                    except Exception as e:
                        continue

                    # 选择缺陷类型
                    defect_name, defect_fn = rng.choice(DEFECT_GENERATORS)
                    # 随机强度：有些缺陷强一些，有些弱一些
                    try:
                        defected_img = defect_fn(img, rng)
                    except Exception as e:
                        print(f"    ⚠️  生成缺陷失败 ({sf}): {e}")
                        continue

                    # 生成文件名：源文件名+_synth_{缺陷类型}_{序号}
                    base, ext = os.path.splitext(sf)
                    out_name = f"{base}_synth_{defect_name}_{generated_count:03d}{ext}"
                    out_path = defect_dir / out_name

                    # 存为JPG保持一致性
                    defected_img.save(str(out_path), quality=95)
                    generated_count += 1
                    total_generated += 1
                    total_by_type[defect_name] += 1
                    attempts += 1

                    if generated_count % 10 == 0:
                        print(f"    ...已生成 {generated_count}/{needed}")

            print(f"    ✅ 生成 {generated_count} 个缺陷 ({view})")

    # 汇总
    print(f"\n{'='*70}")
    print(f"生成完成！")
    print(f"总共生成: {total_generated} 个合成缺陷")
    if total_by_type:
        print(f"缺陷类型分布:")
        for name, count in sorted(total_by_type.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
