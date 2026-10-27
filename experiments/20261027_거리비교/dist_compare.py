# 10-27 중간보고서 Q2(확대하면 나아지나) 답: 3차 현장 시험 400mm(테스트4) vs 300mm(테스트5) 비교
# 둘 다 같은 시편(NO2, 실측 20.020)·같은 조명(아크릴+A4)·같은 노출(1/40, ISO 40)·연속 5장. 300mm로 정하고 본시험 진행
# 측정: 3차 본분석과 같은 국소 50% 경계(local_edges.py 방식), 원본 배열 좌표(EXIF 방향 3이라 실제와 180도 반대)
import glob
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BASE = '../../ETC/261021/'
SETS = {  # 이름: (폴더, 핀 열, Ø20 단 가운데 열, 세로 창). Ø20 단 열은 cols.py로 열별 지름이 평평한 구간을 확인해 정함
    '400mm 테스트3': ('테스트3', range(6250, 6650, 10), range(8330, 8510, 10), (3300, 4350)),
    '400mm 테스트4': ('테스트4', range(6250, 6650, 10), range(8330, 8510, 10), (3300, 4350)),
    '300mm 테스트5': ('테스트5', range(5300, 5800, 10), range(8030, 8330, 10), (3150, 4350)),
    '300mm 본시험 NO2': ('../../field/photos/20261021_원전기계/3차_261021_분류/1_본시험_암막/NO02', range(5300, 5800, 10), range(8030, 8330, 10), (3150, 4350)),
}

def refine(c, e, sgn):
    out = c[e - 60:e - 20] if sgn > 0 else c[e + 21:e + 61]
    ins = c[e + 10:e + 30] if sgn > 0 else c[e - 29:e - 9]
    lo_v, hi_v = np.median(out), np.median(ins)
    h = (lo_v + hi_v) / 2
    lo, hi = e - 20, e + 20
    seg = c[lo:hi]
    s = np.where(np.diff(np.sign(seg - h)) != 0)[0]
    if len(s) == 0: return np.nan, np.nan
    i = s[0] if sgn > 0 else s[-1]
    pos = lo + i + (h - seg[i]) / (seg[i + 1] - seg[i])
    # 경계 폭: 밝기가 바깥-안쪽 차이의 10% -> 90%로 바뀌는 거리(px)
    w = c[e - 50:e + 50]
    t = (w - hi_v) / (lo_v - hi_v)     # 바깥(밝음)=1, 안쪽(어두움)=0
    if sgn < 0: t = t[::-1]
    try:
        a = np.where(t > 0.9)[0]; a = a[a < 50].max()
        b = np.where(t < 0.1)[0]; b = b[b > 50].min()
        wid = b - a
    except ValueError:
        wid = np.nan
    return pos, wid

def height(g, x, y0, y1):
    c = g[y0:y1, x]
    dk = np.r_[0, (c < 100).astype(int), 0]
    st, en = np.where(np.diff(dk) == 1)[0], np.where(np.diff(dk) == -1)[0] - 1
    keep = (st > 0) & (en < len(c) - 1) & (en - st >= 10)
    st, en = st[keep], en[keep]
    if len(st) == 0: return [np.nan] * 3
    t, wt = refine(c, st[0], +1); b, wb = refine(c, en[-1], -1)
    return b - t, wt, wb

rows = {}
for name, (d, px, sx, (y0, y1)) in SETS.items():
    r = []
    for f in sorted(glob.glob((d if d.startswith('../') else BASE + d) + '/*.jpg')):
        g = np.asarray(Image.open(f).convert('L'), dtype=float)
        p = np.array([height(g, x, y0, y1) for x in px]); s = np.array([height(g, x, y0, y1) for x in sx])
        r.append([np.nanmedian(p[:, 0]), np.nanmedian(s[:, 0]), np.nanmedian(np.r_[s[:, 1], s[:, 2]]), np.nanstd(s[:, 0]), np.median(g[y1 - 60:y1, list(sx)])])
        print(name, f.split('/')[-1], ' '.join(f'{v:.2f}' for v in r[-1]), flush=True)
    rows[name] = np.array(r)

print('\n요약 (핀 Ø16 기준 스케일, 5장)')
print('조건 | 장수 | 1px(µm) | Ø20 px | 사진 사이 sd px / µm | 경계 폭(10-90%) px / µm | 열 사이 sd µm | 아래 배경 밝기')
for name, r in rows.items():
    um = 16000 / r[:, 0].mean()
    sd = r[:, 1].std(ddof=1)
    print(f'{name} | {len(r)} | {um:.1f} | {r[:,1].mean():.2f} | {sd:.3f} / {sd*um:.1f} | {r[:,2].mean():.1f} / {r[:,2].mean()*um:.0f} | {r[:,3].mean()*um:.1f} | {r[:,4].mean():.0f}')
