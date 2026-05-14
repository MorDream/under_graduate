#!/usr/bin/env python3
"""生成10张不同基底的"全家福"合成缺陷图"""
import sys
sys.path.insert(0, '.')
from generate_synthetic_defects import imread_unicode, imwrite_unicode, add_scratch, add_stains, add_spots, add_missing_region, add_texture_noise
import numpy as np
from pathlib import Path

out_dir = Path('experiment_data/visualization/synthetic_all_in_one')
out_dir.mkdir(parents=True, exist_ok=True)

# 收集10张不同品类的test/good图
all_good = []
for cat_dir in sorted(Path('data/晶圆分类数据集').iterdir()):
    if not cat_dir.is_dir():
        continue
    good_dir = cat_dir / 'test' / 'good'
    if good_dir.exists():
        for f in sorted(good_dir.glob('*.jpg')):
            all_good.append(f)
            if len(all_good) >= 10:
                break
    if len(all_good) >= 10:
        break

print(f"Found {len(all_good)} base images")
all_good = all_good[:10]

for idx, img_path in enumerate(all_good):
    img = imread_unicode(str(img_path))
    if img is None:
        print(f"  [{idx}] FAIL: {img_path.name}")
        continue

    h, w = img.shape[:2]
    result = img.copy()
    result = add_scratch(result, max_scratches=2)
    result = add_stains(result, max_stains=2)
    result = add_spots(result, max_spots=8)
    result = add_missing_region(result, max_regions=1)
    result = add_texture_noise(result)

    out_path = out_dir / f"alldefects_{idx:02d}_{img_path.stem}.jpg"
    imwrite_unicode(str(out_path), result)
    print(f"  [{idx}] {img_path.stem:40s} -> {out_path.name}")

print(f"\nDone! Saved to {out_dir}")
for f in sorted(out_dir.glob('*')):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")
