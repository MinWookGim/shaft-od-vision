import sys
sys.path.insert(0, "/home/gmw/data/Business/SmallManufacturer/2026/샤프트측정/experiments")
import shaft_core as core
import numpy as np
from dataclasses import replace

core.MODEL_PATH = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/models/demo_yolo11n_20260917.pt"
SRC = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/photos/20260916_원전기계/20260916_154320.jpg"

def points(path, p):
    img = core.imread(path); box, conf = core.detect_box(path)
    x1,y1,x2,y2 = box; box_h = y2-y1
    vm = int(box_h*p.vertical_margin)
    my1, my2 = max(0,y1-vm), min(img.shape[0], y2+vm)
    roi = img[my1:my2, x1:x2]
    blur = core.cv2.GaussianBlur(core.cv2.cvtColor(roi, core.cv2.COLOR_BGR2GRAY),(p.blur_ksize,p.blur_ksize),0)
    roi_h, roi_w = blur.shape
    xs = np.linspace(int(roi_w*p.x_start_ratio), int(roi_w*p.x_end_ratio), p.n_points).astype(int)
    dia_lo, dia_hi = (box_h*p.dia_min_box, box_h*p.dia_max_box) if p.dia_basis=="box" else (None,None)
    T,B = [],[]
    for x in xs:
        r = core.find_outer_edge(blur, x, p, dia_lo, dia_hi)
        if r is None: T.append(np.nan); B.append(np.nan); continue
        T.append(my1+r[0]); B.append(my1+r[1])
    return np.array(T), np.array(B)

base = core.Params()
fixed = replace(base, vertical_margin=0.05, dia_basis="box")

T_ok, B_ok = points(SRC, fixed)      # 기준(정답에 붙은 설정)
T_bad, B_bad = points(SRC, base)     # 기본 설정

ref_top, ref_bot = np.nanmedian(T_ok), np.nanmedian(B_ok)
TOL = 20.0  # 20px(약 0.35mm) 넘게 벗어나면 "다른 데를 잡았다"로 본다

dt = np.abs(T_bad - ref_top); db = np.abs(B_bad - ref_bot)
print(f"기준 상단 y={ref_top:.1f}  하단 y={ref_bot:.1f}  (수정설정 median)")
print(f"기본설정 31점 중 상단이 기준에서 {TOL:.0f}px 넘게 벗어난 점: {int((dt>TOL).sum())}/31   (최대 {np.nanmax(dt):.0f}px)")
print(f"기본설정 31점 중 하단이 기준에서 {TOL:.0f}px 넘게 벗어난 점: {int((db>TOL).sum())}/31   (최대 {np.nanmax(db):.0f}px)")
print(f"상단·하단 둘 다 정상인 점: {int(((dt<=TOL)&(db<=TOL)).sum())}/31")
print()
print("상단 이탈량 분포(px):", np.round(np.sort(dt),0).astype(int).tolist())
print("하단 이탈량 분포(px):", np.round(np.sort(db),0).astype(int).tolist())
