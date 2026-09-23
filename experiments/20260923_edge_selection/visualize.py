# 31개 측정점이 실제로 어디를 찍는지 원본 사진 위에 그려서 눈으로 보이게 한다.
# measure_roi 와 완전히 같은 로직(같은 Params, 같은 MAD 필터)을 그대로 써서
# 집계 숫자만 뽑지 않고 "점 하나하나의 좌표"까지 꺼낸다.
import sys
sys.path.insert(0, "/home/gmw/data/Business/SmallManufacturer/2026/샤프트측정/experiments")
import shaft_core as core
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SRC = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/photos/20260916_원전기계/20260916_154320.jpg"
MODEL = "/home/gmw/Documents/AIFFEL_Work/_scratch/Pre365/field/models/demo_yolo11n_20260917.pt"
core.MODEL_PATH = MODEL
OUT = "/tmp/claude-1000/-home-gmw-Documents-AIFFEL-Work/64e02336-4f92-45ac-aa5b-e4ff473f94d7/scratchpad/verify_31pt"

def measure_with_points(path, p=None):
    """measure_roi 와 같은 절차를 따르되, 31개 점 각각의 (x_abs, top_y_abs, bottom_y_abs, valid?) 를 같이 반환한다."""
    p = p or core.Params()
    img = core.imread(path)
    box, conf = core.detect_box(path)
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    vm = int(box_h * p.vertical_margin)
    my1 = max(0, y1 - vm); my2 = min(img.shape[0], y2 + vm)
    roi = img[my1:my2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
    roi_h, roi_w = blur.shape
    x_positions = np.linspace(int(roi_w*p.x_start_ratio), int(roi_w*p.x_end_ratio), p.n_points).astype(int)
    dia_lo, dia_hi = (box_h*p.dia_min_box, box_h*p.dia_max_box) if p.dia_basis=="box" else (None, None)

    pts = []  # (x_local, top_sub, bottom_sub, diameter)
    for x in x_positions:
        r = core.find_outer_edge(blur, x, p, dia_lo, dia_hi)
        if r is None:
            pts.append((x, None, None, None)); continue
        top_sub, bottom_sub, dia = r
        pts.append((x, top_sub, bottom_sub, dia))

    diam_vals = np.array([q[3] for q in pts if q[3] is not None])
    med = float(np.median(diam_vals))
    dev = np.abs(diam_vals - med); mad = float(np.median(dev))
    sigma = 1.4826*mad if mad>0 else 0

    out_pts = []
    for (x, top, bot, dia) in pts:
        if dia is None:
            out_pts.append(dict(x_abs=x1+x, valid=False, found=False)); continue
        is_valid = (abs(dia-med) <= p.mad_k*sigma) if sigma>0 else True
        out_pts.append(dict(x_abs=x1+x, top_abs=my1+top, bottom_abs=my1+bot, dia=dia, valid=bool(is_valid)))
    return dict(box=box, conf=conf, roi=(x1,my1,x2,my2), pts=out_pts, median=med, std=float(np.std(diam_vals)))

def render(path, result, crop_pad=0.15, dot_r=None, line_w=None, font_size=None, label=""):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    x1,y1,x2,y2 = result["roi"]
    w = x2-x1; h=y2-y1
    dot_r = dot_r or max(2, int(w*0.01))
    line_w = line_w or max(1, int(w*0.004))
    for pt in result["pts"]:
        if not pt.get("found", True):
            continue
        x = pt["x_abs"]
        color_top = (255,60,60) if pt["valid"] else (150,150,150)
        color_bot = (60,120,255) if pt["valid"] else (150,150,150)
        d.line([(x, pt["top_abs"]-dot_r*2), (x, pt["top_abs"]+dot_r*2)], fill=color_top, width=line_w)
        d.line([(x, pt["bottom_abs"]-dot_r*2), (x, pt["bottom_abs"]+dot_r*2)], fill=color_bot, width=line_w)
    # crop
    pad_x = int(w*crop_pad); pad_y = int(h*0.1)
    cx1, cy1 = max(0,x1-pad_x), max(0,y1-pad_y)
    cx2, cy2 = min(im.width, x2+pad_x), min(im.height, y2+pad_y)
    crop = im.crop((cx1,cy1,cx2,cy2))
    return crop

def std_of(pts, only_valid):
    import numpy as np
    vals = [p["dia"] for p in pts if p.get("dia") is not None and (p["valid"] or not only_valid)]
    return float(np.std(vals)) if vals else float("nan")

r_orig = measure_with_points(SRC)
n_valid_o = sum(1 for p in r_orig["pts"] if p.get("valid"))
print("원본:", "box", r_orig["box"], "conf", round(r_orig["conf"],2), "median", round(r_orig["median"],1))
print(f"  31점 전체 std(이상치 포함) = {std_of(r_orig['pts'], False):.2f}px | 유효 {n_valid_o}/31 만의 std = {std_of(r_orig['pts'], True):.3f}px")

# 2992 축소본
small_path = OUT + "/a2_2992.jpg"
if not __import__("os").path.exists(small_path):
    im0 = Image.open(SRC).convert("RGB"); w0,h0=im0.size; s=2992/max(w0,h0)
    im0.resize((round(w0*s), round(h0*s)), Image.LANCZOS).save(small_path, quality=95)
r_small = measure_with_points(small_path)
n_valid_s = sum(1 for p in r_small["pts"] if p.get("valid"))
print("2992:", "box", r_small["box"], "conf", round(r_small["conf"],2), "median", round(r_small["median"],1))
print(f"  31점 전체 std(이상치 포함) = {std_of(r_small['pts'], False):.2f}px | 유효 {n_valid_s}/31 만의 std = {std_of(r_small['pts'], True):.3f}px")

crop_o = render(SRC, r_orig)
crop_s = render(small_path, r_small)

# 같은 물리적 화면 배율로 보이게, 두 crop을 같은 세로 높이로 리사이즈해서 나란히
target_h = 900
def fit(im, h):
    w = int(im.width * h/im.height)
    return im.resize((w,h), Image.LANCZOS)
col_w = 420
def fit_w(im, w):
    h = int(im.height * w/im.width)
    return im.resize((w,h), Image.LANCZOS)
crop_o2 = fit_w(crop_o, col_w)
crop_s2 = fit_w(crop_s, col_w)
target_h = max(crop_o2.height, crop_s2.height)

gap=40
head_h = 90
foot_h = 50
canvas = Image.new("RGB", (col_w*2+gap, target_h+head_h+foot_h), (255,255,255))
d = ImageDraw.Draw(canvas)
canvas.paste(crop_o2, (0,head_h))
canvas.paste(crop_s2, (col_w+gap,head_h))
try:
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20, index=2)
    font_s = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 16, index=2)
except Exception:
    font = font_s = ImageFont.load_default()
d.text((10,8), "원본 6008px", fill=(0,0,0), font=font)
d.text((10,34), f"유효 {n_valid_o}/31", fill=(0,0,0), font=font_s)
d.text((10,56), f"유효점만 std={std_of(r_orig['pts'],True):.1f}px", fill=(0,0,0), font=font_s)
d.text((col_w+gap+10,8), "2992px 축소", fill=(0,0,0), font=font)
d.text((col_w+gap+10,34), f"유효 {n_valid_s}/31", fill=(0,0,0), font=font_s)
d.text((col_w+gap+10,56), f"유효점만 std={std_of(r_small['pts'],True):.2f}px", fill=(0,0,0), font=font_s)
d.text((10, head_h+target_h+14), "빨강=상단엣지  파랑=하단엣지  회색=이상치로 걸러진 점", fill=(60,60,60), font=font_s)
canvas.save(OUT + "/A2_31점_비교.png")
print("저장:", OUT + "/A2_31점_비교.png")
