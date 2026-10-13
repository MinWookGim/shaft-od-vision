# 10-08 현장: 윗면 밝은 띠 때문에 50% 경계가 밀리는지, 바깥 경계(배경 -> 물체)로 잡으면 안정적인지 비교
import sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def cross(p, thr, y0, y1, step):
    # p[y0] 부터 step 방향으로 가며 thr 아래로 처음 내려가는 위치(소수점 보간)
    y = y0
    while y != y1:
        a, b = p[y], p[y + step]
        if a >= thr > b:
            return y + step * (a - thr) / (a - b)
        y += step
    return np.nan

def widths(g, x0, x1, y0, y1, bg, frac):
    out = []
    for x in range(x0, x1):
        p = g[y0:y1, x]
        thr = bg * frac + p[(y1-y0)//2] * (1 - frac)          # 배경과 가운데(어두운 몸통) 사이 문턱
        top = cross(p, thr, 0, len(p) - 1, 1)                  # 위에서 아래로
        bot = cross(p, thr, len(p) - 1, 0, -1)                 # 아래에서 위로
        out.append((len(p) - 1 - 0) * 0 + (y1 - y0 - 1 - (len(p) - 1 - bot)) - top if not np.isnan(top+bot) else np.nan)
    w = np.array(out)
    return np.nanmedian(w), np.nanstd(w)

for f in sys.argv[1:]:
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    bg = g[1000:2000, 6000:7000].mean()
    r = [f.split('/')[-1]]
    for name, roi in (('샤프트', (8350, 9150, 3550, 4400)), ('핀', (9585, 9665, 3700, 4250))):
        for frac in (0.5, 0.85):                               # 0.5 = 지금 방식, 0.85 = 바깥 경계 쪽
            m, s = widths(g, *roi, bg, frac)
            r.append(f'{name}{int(frac*100)}% {m:7.2f}±{s:4.2f}')
    print('  '.join(r))
