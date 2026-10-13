# 10-08 t1: 시편 묶음별 샤프트 중심 높이·폭, 핀 폭(폰 움직임), 경계 선명도(초점) 비교
import sys, glob, os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def edges(col, thr):
    # 위에서 아래로 처음 thr 아래로 내려가는 점(위 경계), 아래에서 위로 처음 내려가는 점(아래 경계). 소수점 보간
    below = col < thr
    if not below.any(): return np.nan, np.nan
    i = np.argmax(below); j = len(col) - 1 - np.argmax(below[::-1])
    top = i - 1 + (col[i-1] - thr) / (col[i-1] - col[i])
    bot = j + (col[j] - thr) / (col[j] - col[j+1])
    return top, bot

def ramp(col, y, lo, hi, up):
    # 경계 y 주변 40px 안에서 10%~90% 사이를 지나는 길이(px) = 흐림 정도
    a = col[int(y)-40:int(y)+41]
    t10 = lo + .1*(hi-lo); t90 = lo + .9*(hi-lo)
    return np.sum((a > t10) & (a < t90))

def meas(g, x0, x1, y0, y1, bg):
    tops, bots, rt, rb = [], [], [], []
    for x in range(x0, x1, 2):
        col = g[y0:y1, x]
        dark = np.percentile(col, 5)
        thr = (bg + dark) / 2
        t, b = edges(col, thr)
        if np.isnan(t): continue
        tops.append(t + y0); bots.append(b + y0)
        rt.append(ramp(col, t, dark, bg, True)); rb.append(ramp(col, b, dark, bg, False))
    tops, bots = np.array(tops), np.array(bots)
    w = bots - tops
    return np.median(w), np.std(w), np.median((tops+bots)/2), np.median(rt), np.median(rb)

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(sys.argv[1] + '/*.jpg')):
        g = np.asarray(Image.open(f).convert('L'), dtype=float)
        bgroi = g[4500:4800, 6800:7600]
        bg = bgroi.mean(); sat = (bgroi >= 254).mean() * 100
        sw, ssd, sc, st, sb = meas(g, 8350, 9150, 3560, 4400, bg)
        pw, psd, pc, pt, pb = meas(g, 9655, 9700, 3700, 4250, bg)
        t = os.path.basename(f)[9:15]
        rows.append((t, sw, ssd, sc, st, sb, pw, psd, pc, bg, sat))
        print(f'{t}  샤프트폭 {sw:7.2f} sd{ssd:5.2f} 중심y {sc:7.1f} 흐림(위/아래) {st:3.0f}/{sb:3.0f}  핀폭 {pw:6.2f} sd{psd:4.2f} 핀중심 {pc:7.1f}  배경 {bg:5.1f} 포화 {sat:4.1f}%')
    np.save(sys.argv[1] + '/../_out/t1_rows.npy', np.array(rows, dtype=object))
