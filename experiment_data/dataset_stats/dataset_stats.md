
## MVTec AD 数据集统计

| 品类 | 类型 | train/good | test/good | test/defect | test总计 | 缺陷类型数 |
|------|------|------------|-----------|-------------|----------|-----------|
| bottle       | 物体   |        209 |        20 |          63 |       83 |         3 |
| cable        | 物体   |        224 |        58 |          92 |      150 |         8 |
| capsule      | 物体   |        219 |        23 |         109 |      132 |         5 |
| carpet       | 纹理   |        280 |        28 |          89 |      117 |         5 |
| grid         | 纹理   |        264 |        21 |          57 |       78 |         5 |
| hazelnut     | 物体   |        391 |        40 |          70 |      110 |         4 |
| leather      | 纹理   |        245 |        32 |          92 |      124 |         5 |
| metal_nut    | 物体   |        220 |        22 |          93 |      115 |         4 |
| pill         | 物体   |        267 |        26 |         141 |      167 |         7 |
| screw        | 物体   |        320 |        41 |         119 |      160 |         5 |
| tile         | 纹理   |        230 |        33 |          84 |      117 |         5 |
| toothbrush   | 物体   |         60 |        12 |          30 |       42 |         1 |
| transistor   | 物体   |        213 |        60 |          40 |      100 |         4 |
| wood         | 纹理   |        247 |        19 |          60 |       79 |         5 |
| zipper       | 物体   |        240 |        32 |         119 |      151 |         7 |
| **合计** | | **3629** | **467** | **1258** | **1725** | |

### 各品类缺陷类型详情

- **bottle**: broken_large(20), broken_small(22), contamination(21)
- **cable**: bent_wire(13), cable_swap(12), combined(11), cut_inner_insulation(14), cut_outer_insulation(10), missing_cable(12), missing_wire(10), poke_insulation(10)
- **capsule**: crack(23), faulty_imprint(22), poke(21), scratch(23), squeeze(20)
- **carpet**: color(19), cut(17), hole(17), metal_contamination(17), thread(19)
- **grid**: bent(12), broken(12), glue(11), metal_contamination(11), thread(11)
- **hazelnut**: crack(18), cut(17), hole(18), print(17)
- **leather**: color(19), cut(19), fold(17), glue(19), poke(18)
- **metal_nut**: bent(25), color(22), flip(23), scratch(23)
- **pill**: color(25), combined(17), contamination(21), crack(26), faulty_imprint(19), pill_type(9), scratch(24)
- **screw**: manipulated_front(24), scratch_head(24), scratch_neck(25), thread_side(23), thread_top(23)
- **tile**: crack(17), glue_strip(18), gray_stroke(16), oil(18), rough(15)
- **toothbrush**: defective(30)
- **transistor**: bent_lead(10), cut_lead(10), damaged_case(10), misplaced(10)
- **wood**: color(8), combined(11), hole(10), liquid(10), scratch(21)
- **zipper**: broken_teeth(19), combined(16), fabric_border(17), fabric_interior(16), rough(17), split_teeth(18), squeezed_teeth(16)

## 晶圆分类数据集统计

| 品类 | train/good | 视图 | test/good | test/defect |
|------|------------|------|-----------|-------------|
| BGA 12x4           |        151 | UP   |        26 |          26 |
| BGA 12x4           |        151 | DOWN |        30 |          30 |
| BGA S5E 16x7       |        322 | UP   |         9 |           9 |
| BGA S5E 16x7       |        322 | DOWN |         7 |           7 |
| ESSD 12x4          |        167 | UP   |         7 |           5 |
| ESSD 12x4          |        167 | DOWN |         3 |           5 |
| ESSD 12x5          |        469 | UP   |         4 |           5 |
| ESSD 12x5          |        469 | DOWN |         6 |           5 |
| INAND 19x5         |        168 | UP   |         7 |           5 |
| INAND 19x5         |        168 | DOWN |         9 |          11 |
| MicroSD 20x4       |        318 | UP   |         6 |           5 |
| MicroSD 20x4       |        318 | DOWN |         4 |           5 |
| SDSIP 22x3         |        239 | UP   |         5 |           5 |
| SDSIP 22x3         |        239 | DOWN |         5 |           5 |
| UBGA 12x5          |        206 | UP   |         5 |           5 |
| UBGA 12x5          |        206 | DOWN |         5 |           5 |
| **UP视图合计** | **2040** | | **69** | **65** |
| **DOWN视图合计** | **2040** | | **69** | **73** |
