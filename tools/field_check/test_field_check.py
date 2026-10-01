# field_check.py 검증: 정답을 아는 합성 사진으로 숫자가 맞는지 본다
import sys, os, json, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_check import measure_width, edges_in_profile, apparent

H, W, SS = 1200, 1600, 20  # SS = 슈퍼샘플링 배수(소수점 폭을 흉내 내려고 크게 그려서 줄인다)

def synth(shaft_w, pin_w, shift=0.0, bg=230, dark=40, blur=3.0, bg_left=None):
    big = np.full((H * SS, W * SS), bg, np.float32)
    # 샤프트: 가로로 누운 띠, 세로 폭 shaft_w px, 위 경계 y=400+shift
    y0 = (400 + shift) * SS
    big[int(round(y0)):int(round(y0 + shaft_w * SS)), 100 * SS:900 * SS] = dark
    # 세운 핀: 세로 띠, 가로 폭 pin_w px, 왼쪽 경계 x=1200+shift
    x0 = (1200 + shift) * SS
    big[300 * SS:900 * SS, int(round(x0)):int(round(x0 + pin_w * SS))] = dark
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), blur)  # 렌즈 흐림 흉내
    if bg_left is not None:
        img[300:900, 1100:1130] = bg_left  # 핀 ROI 왼쪽 끝에 어두운 물체가 걸린 상황
    return np.clip(img, 0, 255).astype(np.uint8)

cfg_shaft, cfg_pin = [200, 800, 250, 1000], [1100, 1550, 400, 800]
ok = True
def check(name, got, want, tol):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"{'OK ' if good else 'NG '} {name}: {got:.3f} (정답 {want:.3f}, 허용 ±{tol})")

# 1) 소수점 폭을 맞히나
for sw, pw in ((500.0, 250.0), (503.35, 251.7), (420.8, 252.25)):
    img = synth(sw, pw)
    check(f'샤프트 {sw}', measure_width(img, cfg_shaft, 'v')['width_px'], sw, 0.1)
    check(f'핀(가로) {pw}', measure_width(img, cfg_pin, 'h')['width_px'], pw, 0.1)

# 2) 위치만 0.37px 옮겨도 폭은 그대로여야 한다 (A4 에서 '폭 변화'만 보려는 것)
a = measure_width(synth(500.0, 250.0), cfg_pin, 'h')['width_px']
b = measure_width(synth(500.0, 250.0, shift=0.37), cfg_pin, 'h')['width_px']
check('위치 이동 후 핀 폭 차이', b - a, 0.0, 0.05)

# 3) 0.1% 바뀐 핀(250 -> 250.25)을 구별하나 (A4 문턱)
c = measure_width(synth(500.0, 250.25), cfg_pin, 'h')['width_px']
check('핀 0.1% 변화 검출', (c - a) / a * 100, 0.1, 0.02)

# 4) ROI 한쪽 끝이 어두울 때 (집 사진에서 고친 버그)
img = synth(500.0, 250.0, bg_left=120)
check('한쪽 배경 어두움 핀 폭', measure_width(img, cfg_pin, 'h')['width_px'], 250.0, 0.1)

# 5) A5 계산: 샤프트 20.020 / 핀 10.000 을 D=400 에서 '보이는 대로' px 로 만들고 환산이 실측으로 돌아오나
D, k = 400.0, 25.0  # k = px/mm
sw, pw = apparent(20.020, D) * k, apparent(10.0, D) * k
import tempfile
d = tempfile.mkdtemp()  # 합성 사진은 임시 폴더에 (repo 안에 안 남게)
cv2.imwrite(f'{d}/syn_a5.png', synth(sw, pw))
cv2.imwrite(f'{d}/syn_a5_far.png', synth(sw, pw * (1 - 0.002)))  # 핀이 0.2% 작게 = 핀이 0.8mm 멀리
json.dump({'shaft': {'roi': cfg_shaft}, 'pin': {'roi': cfg_pin, 'axis': 'h'}, 'bg': {'roi': [100, 1500, 20, 200]},
           'true_shaft_mm': 20.020, 'pin_mm': 10.0, 'D_mm': D}, open(f'{d}/syn.json', 'w'))
out = subprocess.run([sys.executable, os.path.join(sys.path[0], 'field_check.py'), 'check', f'{d}/syn.json',
                      f'{d}/syn_a5.png', f'{d}/syn_a5_far.png'], capture_output=True, text=True).stdout
print(out)
print('전체:', '통과' if ok else '실패 있음')
