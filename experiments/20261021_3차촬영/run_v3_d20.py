# V3 재실행: 측정 구간을 Ø20 자리(칼라와 홈 사이 짧은 단)로 바로잡음.
# 앞 run_v3.py 는 끝단(약 Ø16.7, x 8900~9900)을 잰 실수 -> 결과는 v3_px.csv 로 남겨 둠(지우지 않음)
# 시편마다 축 방향 위치가 조금씩 달라서 사진마다 칼라 왼쪽 끝과 홈 오른쪽 끝을 찾아 상자를 잡는다
# 같은 패스에서 내 50% 방식 지름(같은 구간)과 핀 Ø16 높이도 같이 잰다
import sys, csv
from pathlib import Path
import numpy as np
from PIL import Image
from pos_scan_lib import cross
sys.path.insert(0, '../../build/demo/hy_system/measurement')
import measurement_engine_EDGE_V3_REFLECTION as v3
Image.MAX_IMAGE_PIXELS = None
D = Path('../../field/photos/20261021_원전기계/3차_261021')
NO1_OFF = (2408, 1112)          # NO1 은 1.41배 줌으로 가운데 잘림 -> 좌표를 이만큼 빼서 씀
Y0, Y1 = 3000, 4600             # 원본(줌 없는) 좌표에서 샤프트를 감싸는 세로 창
PIN = (5350, 5800)              # 핀 Ø16 이 V블록에 안 가리고 보이는 가로 구간
cal = {'a': 1.0, 'b': 0.0}

def tb(g, x, y0, y1):
    # 한 열의 위·아래 50% 경계(서브픽셀)
    col = g[y0:y1, x]; n = len(col) // 2
    t = cross(col[:n]); b = cross(col[n:][::-1])
    return y0 + t, y0 + (n - 1 - b) + n

def height_scan(g, xs, y0, y1):
    return np.array([np.subtract(*tb(g, x, y0, y1)[::-1]) for x in xs])

def locate(g, dx, dy):
    # 높이를 5px 간격으로 재고 중앙값 필터(9칸)로 홈·잡신호를 누른 뒤
    # Ø20 단 높이(lev) = x 8100~8200 중앙값, 칼라 왼쪽 끝 cL = lev+40 을 처음 넘는 곳, 홈 쪽 끝 gR = lev 에서 15 넘게 벗어나는 마지막 곳
    xs = np.arange(7800, 8700, 5) - dx
    h = height_scan(g, xs, Y0 - dy, Y1 - dy)
    hf = np.array([np.nanmedian(h[max(0, i - 4):i + 5]) for i in range(len(h))])
    mid = (xs >= 8100 - dx) & (xs <= 8200 - dx)
    lev = np.nanmedian(hf[mid])
    i0 = np.where(mid)[0][0]
    j = i0
    while j < len(xs) - 1 and hf[j] < lev + 40: j += 1
    cL = xs[j]
    k = i0
    while k > 0 and abs(hf[k] - lev) < 15: k -= 1
    gR = xs[k + 1]
    if cL - gR < 200: raise ValueError(f'구간 너무 짧음 {gR}~{cL}')
    return gR, cL, lev

out = open('v3_d20.csv', 'w', newline=''); w = csv.writer(out)
w.writerow(['file', 'x_groove', 'x_collar', 'box_x0', 'box_x1', 'v3_px', 'v3_mean', 'v3_std', 'v3_valid',
            'p50_px', 'p50_sd', 'pin_px', 'pin_sd'])
for f in sorted(D.glob('2026*_??????.jpg')):
    dx, dy = NO1_OFF if '_1533' in f.name else (0, 0)
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    try:
        # 위치 탐색(locate) 시험 결과 홈 끝 약 7995, 칼라 끝 약 8355 로 세트 상관없이 거의 고정 -> 고정 상자(양끝 20px 여유)
        gR, cL = 7995 - dx, 8355 - dx
        x0, x1 = gR + 20, cL - 20
        hs = height_scan(g, np.arange(x0, x1, 4), Y0 - dy, Y1 - dy)
        ts = [tb(g, x, Y0 - dy, Y1 - dy) for x in range(x0, x1, 20)]
        top, bot = np.nanmedian([t for t, b in ts]), np.nanmedian([b for t, b in ts])
        ph = height_scan(g, np.arange(PIN[0] - dx, PIN[1] - dx, 5), Y0 - dy, Y1 - dy)
        box = (x0, int(top - 90), x1, int(bot + 90))
        r = v3.measure(f, box, float('nan'), cal, Path('v3_out_d20'))
        w.writerow([f.name, gR, cL, x0, x1, r['measured_pixel'], r['mean_pixel'], r['std_pixel'], r['valid_point_count'],
                    np.nanmedian(hs), np.nanstd(hs), np.nanmedian(ph), np.nanstd(ph)])
    except Exception as e:
        w.writerow([f.name] + [''] * 7 + ['ERR ' + str(e)[:60]] + [''] * 4)
    out.flush(); print(f.name, flush=True)
