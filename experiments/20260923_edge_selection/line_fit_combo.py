# 다겸님 제안(31개 점은 직선 위에 있어야 한다) + 희연님이 찾아준 원조 코드(Day026 line_fitting_measurement.py)를
# 오늘 수정(margin+dia_basis=box)과 같이 써 봤을 때 어떻게 되는지 직접 확인.
from paths import load_core, photo_dir, A2_NAME
core = load_core()
import numpy as np
from dataclasses import replace

DIR = str(photo_dir()) + "/"

def raw_points(path, p):
    """measure_roi 와 같은 절차로 31개 점의 (x, top_sub, bottom_sub) 원자료를 뽑는다. MAD 필터는 안 건다."""
    img = core.imread(path)
    box, conf = core.detect_box(path)
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    vm = int(box_h * p.vertical_margin)
    my1, my2 = max(0, y1 - vm), min(img.shape[0], y2 + vm)
    roi = img[my1:my2, x1:x2]
    gray = core.cv2.cvtColor(roi, core.cv2.COLOR_BGR2GRAY)
    blur = core.cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
    roi_h, roi_w = blur.shape
    xs = np.linspace(int(roi_w*p.x_start_ratio), int(roi_w*p.x_end_ratio), p.n_points).astype(int)
    dia_lo, dia_hi = (box_h*p.dia_min_box, box_h*p.dia_max_box) if p.dia_basis=="box" else (None, None)
    tops, bottoms, xs_ok = [], [], []
    for x in xs:
        r = core.find_outer_edge(blur, x, p, dia_lo, dia_hi)
        if r is None:
            continue
        top, bot, dia = r
        xs_ok.append(x); tops.append(top); bottoms.append(bot)
    return np.array(xs_ok, float), np.array(tops, float), np.array(bottoms, float), box_h

def robust_line_fit(xs, ys, k=3.0, max_iter=3):
    """직선을 긋고, 직선에서 k*MAD 넘게 벗어난 점을 빼고 다시 긋기를 반복한다(다겸님 제안)."""
    mask = np.ones(len(xs), dtype=bool)
    for _ in range(max_iter):
        a, b = np.polyfit(xs[mask], ys[mask], 1)
        resid = ys - (a*xs + b)
        mad = np.median(np.abs(resid[mask] - np.median(resid[mask])))
        sigma = 1.4826*mad if mad > 0 else 1e-9
        new_mask = np.abs(resid) <= k*sigma
        if new_mask.sum() < 5 or (new_mask == mask).all():
            mask = new_mask; break
        mask = new_mask
    a, b = np.polyfit(xs[mask], ys[mask], 1)
    return a, b, mask

def eval_line_fit(path, p, label):
    xs, tops, bottoms, box_h = raw_points(path, p)
    ta, tb, tmask = robust_line_fit(xs, tops)
    ba, bb, bmask = robust_line_fit(xs, bottoms)
    diam_line = (ba*xs+bb) - (ta*xs+tb)   # 직선으로 계산한 지름(전 구간)
    print(f"{label:38s} 상단기울기={ta:+.4f}(제외{(~tmask).sum()}) 하단기울기={ba:+.4f}(제외{(~bmask).sum()}) "
          f"직선지름 median={np.median(diam_line):8.2f}px std={np.std(diam_line):7.3f}px  (n={len(xs)})")

base = core.Params()
fixed = replace(base, vertical_margin=0.05, dia_basis="box")

SRC = DIR + A2_NAME  # A2
print("=== A2, 원본 6008px ===")
eval_line_fit(SRC, base,  "① 기본설정 + 라인피팅만(둘 다 안 바꿈)")
eval_line_fit(SRC, fixed, "② margin+box 수정 + 라인피팅까지 추가")

print()
print("=== 시편1 14장 전체, 원본 6008px, 수정설정(margin+box)에 라인피팅까지 추가 ===")
files = [
    (1,"20260916_154318.jpg"),(2,"20260916_154320.jpg"),(3,"20260916_154321.jpg"),
    (4,"20260916_154322.jpg"),(5,"20260916_154324.jpg"),(6,"20260916_154325.jpg"),
    (7,"20260916_154326.jpg"),(8,"20260916_154332.jpg"),(9,"20260916_154333.jpg"),
    (10,"20260916_154334.jpg"),(11,"20260916_154336.jpg"),(12,"20260916_154337.jpg"),
    (13,"20260916_154338.jpg"),(14,"20260916_154339.jpg"),
]
meds = []
print(f"{'장':>3} | {'박스기준만(MAD)':>16} | {'박스기준+라인피팅':>18}")
for n, fn in files:
    path = DIR + fn
    r_mad = core.measure_roi(core.imread(path), core.detect_box(path)[0], fixed)
    m_mad = f"{r_mad['measured_pixel']:.2f}({r_mad['std_pixel']:.2f})" if r_mad["ok"] else r_mad["reason"]
    xs, tops, bottoms, box_h = raw_points(path, fixed)
    ta, tb, tmask = robust_line_fit(xs, tops)
    ba, bb, bmask = robust_line_fit(xs, bottoms)
    diam_line = (ba*xs+bb) - (ta*xs+tb)
    med_l, std_l = np.median(diam_line), np.std(diam_line)
    meds.append(med_l)
    print(f"{n:>3} | {m_mad:>16} | {med_l:8.2f}({std_l:.3f})")

meds = np.array(meds)
print()
print(f"박스기준+라인피팅으로 14장 장간 반복 sd = {meds.std():.3f}px (평균 {meds.mean():.2f}px)")
