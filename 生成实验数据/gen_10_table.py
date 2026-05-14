#!/usr/bin/env python3
"""生成10张基底图的"缺陷类型对比表"（原图 + 各缺陷分开展示）"""
import sys
sys.path.insert(0, '.')
from generate_synthetic_defects import imread_unicode, imwrite_unicode, add_scratch, add_stains, add_spots, add_missing_region, add_texture_noise, generate_defect
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# === 中文字体 ===
def get_font(size=20):
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/mnt/c/Windows/Fonts/msyh.ttc',
        '/mnt/c/Windows/Fonts/simhei.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    return ImageFont.load_default()

font_title = get_font(22)
font_label = get_font(16)

def make_defect_table(img_bgr, output_path):
    """生成一张缺陷对比表（2行x3列）"""
    h, w = img_bgr.shape[:2]
    
    # 生成各类型
    variants = [
        ('原图', img_bgr),
        ('划痕', add_scratch(img_bgr.copy())),
        ('污渍', add_stains(img_bgr.copy())),
        ('斑点', add_spots(img_bgr.copy())),
        ('缺失区域', add_missing_region(img_bgr.copy())),
        ('组合缺陷', generate_defect(img_bgr.copy())),
    ]
    
    # 每格大小
    cell_w = w + 20
    cell_h = h + 40
    cols, rows = 3, 2
    
    canvas_w = cell_w * cols + 10
    canvas_h = cell_h * rows + 50
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    
    for idx, (name, img_bgr) in enumerate(variants):
        col = idx % cols
        row = idx // cols
        
        x = 10 + col * cell_w
        y = 50 + row * cell_h
        
        # 标题
        draw.text((x + w//2 - len(name)*8, y + 5), name, fill='black', font=font_label)
        
        # 图片 BGR→RGB→PIL
        img_rgb = img_bgr[..., ::-1]
        pil_img = Image.fromarray(img_rgb)
        canvas.paste(pil_img, (x + 10, y + 30))
    
    # 大标题
    draw.text((10, 5), f'缺陷类型对比 — {Path(output_path).stem}', fill='black', font=font_title)
    
    canvas.save(output_path, quality=95)

# === MAIN ===
out_dir = Path('experiment_data/visualization/synthetic_table')
out_dir.mkdir(parents=True, exist_ok=True)

# 收集10张不同基底的图
all_good = []
for cat_dir in sorted(Path('data/晶圆分类数据集').iterdir()):
    if not cat_dir.is_dir():
        continue
    good_dir = cat_dir / 'test' / 'good'
    if good_dir.exists():
        for f in sorted(good_dir.glob('*'))[:2]:
            all_good.append(f)
            if len(all_good) >= 10:
                break
    if len(all_good) >= 10:
        break

all_good = all_good[:10]
print(f"Base images: {len(all_good)}")

for idx, img_path in enumerate(all_good):
    img = imread_unicode(str(img_path))
    if img is None:
        print(f"  [{idx}] FAIL: {img_path.name}")
        continue
    
    out_name = f"defect_table_{idx:02d}_{img_path.stem}.jpg"
    out_path = out_dir / out_name
    make_defect_table(img, str(out_path))
    print(f"  [{idx}] {out_name}  ({img.shape[1]}x{img.shape[0]})")

print(f"\nDone! {len(all_good)} tables saved to {out_dir}")
