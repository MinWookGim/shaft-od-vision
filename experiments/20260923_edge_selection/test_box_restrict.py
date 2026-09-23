import sys
sys.path.insert(0, "/home/gmw/data/Business/SmallManufacturer/2026/샤프트측정/experiments")
import shaft_core as core
import numpy as np
from dataclasses import replace

SRC = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/photos/20260916_원전기계/20260916_154320.jpg"
MODEL = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/models/demo_yolo11n_20260917.pt"
core.MODEL_PATH = MODEL

img = core.imread(SRC)
box, conf = core.detect_box(SRC)
print("YOLO box", box, "conf", round(conf,2), "box_h", box[3]-box[1])

def try_params(label, p):
    r = core.measure_roi(img, box, p)
    if not r["ok"]:
        print(f"{label:35s} -> 실패: {r['reason']}"); return
    print(f"{label:35s} -> n_valid={r['n_valid']:2d}/{r['n_found']:2d}  median={r['measured_pixel']:8.2f}px  std={r['std_pixel']:8.3f}px")

base = core.Params()
try_params("① 기본(roi기준, margin 0.40)", base)
try_params("② dia_basis=box (범위만 좁힘)", replace(base, dia_basis="box"))
try_params("③ margin 0.40→0.05 (탐색영역 자체를 좁힘)", replace(base, vertical_margin=0.05))
try_params("④ margin 0.05 + dia_basis=box (둘 다)", replace(base, vertical_margin=0.05, dia_basis="box"))
try_params("⑤ margin 0.0 (박스 딱 그대로)", replace(base, vertical_margin=0.0))
