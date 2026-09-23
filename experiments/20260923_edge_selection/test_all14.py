import sys
sys.path.insert(0, "/home/gmw/data/Business/SmallManufacturer/2026/샤프트측정/experiments")
import shaft_core as core
from dataclasses import replace

core.MODEL_PATH = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/models/demo_yolo11n_20260917.pt"
DIR = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/photos/20260916_원전기계/"
files = [
    (1,"20260916_154318.jpg"),(2,"20260916_154320.jpg"),(3,"20260916_154321.jpg"),
    (4,"20260916_154322.jpg"),(5,"20260916_154324.jpg"),(6,"20260916_154325.jpg"),
    (7,"20260916_154326.jpg"),(8,"20260916_154332.jpg"),(9,"20260916_154333.jpg"),
    (10,"20260916_154334.jpg"),(11,"20260916_154336.jpg"),(12,"20260916_154337.jpg"),
    (13,"20260916_154338.jpg"),(14,"20260916_154339.jpg"),
]
ANOM = {2,3,4,5,14}  # D-001에 이미 튄다고 기록된 장번호

base = core.Params()
fixed = replace(base, vertical_margin=0.05, dia_basis="box")

print(f"{'장':>3} {'D-001튐':>7} | {'기본 median':>11} {'기본 std':>9} | {'수정 median':>11} {'수정 std':>9}")
meds_fixed = []
for n, fn in files:
    path = DIR + fn
    img = core.imread(path)
    box, conf = core.detect_box(path)
    r0 = core.measure_roi(img, box, base)
    r1 = core.measure_roi(img, box, fixed)
    m0 = f"{r0['measured_pixel']:.1f}" if r0["ok"] else r0["reason"]
    s0 = f"{r0['std_pixel']:.2f}" if r0["ok"] else "-"
    m1 = f"{r1['measured_pixel']:.1f}" if r1["ok"] else r1["reason"]
    s1 = f"{r1['std_pixel']:.2f}" if r1["ok"] else "-"
    if r1["ok"]:
        meds_fixed.append(r1['measured_pixel'])
    tag = "튐" if n in ANOM else ""
    print(f"{n:>3} {tag:>7} | {m0:>11} {s0:>9} | {m1:>11} {s1:>9}")

import numpy as np
meds_fixed = np.array(meds_fixed)
print()
print(f"수정 설정으로 14장 median 값들의 장간 반복 sd = {meds_fixed.std():.2f}px  (평균 {meds_fixed.mean():.1f}px)")
