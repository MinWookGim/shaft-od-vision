# A2(S01 2번째, 20260916_154320.jpg) 원본(6008) vs 2992 축소본에서
# 31개 X 위치 측정의 "점간 산포"(std_pixel) 자체가 줄어드는지 직접 확인.
# 대표값(median)만 보지 않고 shaft_core.measure_roi 가 반환하는 std_pixel 을 직접 본다.
from paths import load_core, photo
core = load_core()
from PIL import Image
import tempfile
from pathlib import Path
import numpy as np

SRC = str(photo())


def run(path, label):
    img = core.imread(path)
    box, conf = core.detect_box(path)
    if box is None:
        print(label, "YOLO 검출 실패"); return None
    r = core.measure_roi(img, box, core.Params())
    if not r["ok"]:
        print(label, "측정 실패:", r["reason"]); return None
    print(f"{label:20s} size={img.shape[1]}x{img.shape[0]} box={box} conf={conf:.2f} "
          f"n_valid={r['n_valid']}/{r['n_found']} median={r['measured_pixel']:.2f}px "
          f"mean={r['mean_pixel']:.2f}px  std_pixel(31점 산포)={r['std_pixel']:.3f}px")
    return r

print("=== A2 (시편1 2번째, 20260916_154320.jpg) ===")
r_orig = run(SRC, "원본 6008px")

# 2992 축소본 생성 (팀 파이프라인과 동일한 방식: LANCZOS, 긴 변 2992)
small_path = str(Path(tempfile.gettempdir()) / "a2_2992.jpg")   # 중간 산출물이라 임시 폴더에
im = Image.open(SRC).convert("RGB")
w, h = im.size
s = 2992 / max(w, h)
im.resize((round(w*s), round(h*s)), Image.LANCZOS).save(small_path, quality=95)
r_small = run(small_path, "2992px 축소본")

if r_orig and r_small:
    print()
    print("배율 보정(같은 물리 단위 비교용, px * s 로 스케일 맞춤):")
    print(f"  원본 std_pixel 을 2992 스케일로 환산: {r_orig['std_pixel']*s:.3f}px  (2992 기준)")
    print(f"  2992 축소본 std_pixel 그대로       : {r_small['std_pixel']:.3f}px")
