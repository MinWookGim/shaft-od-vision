# 핀이 '배율'이 아니라 '경계가 안쪽으로 먹혀서' 줄었는지 보기: 핀·샤프트 위·아래 경계의 10~90% 폭(흐림 정도)과
# 경계 근처 밝은 쪽·어두운 쪽 밝기를 사진마다 측정. 열마다 3px 평균 단면, 핀 40열·샤프트 21열(튀는 열 뺀 구간) 중앙값
import glob, os
import numpy as np, pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
Y0, Y1 = 3150, 4350
OBJ = {'pin': range(5400, 5800, 10), 'sh': range(8069, 8263, 10)}
def prof(c, e, sgn):
    # sgn +1: 위 경계(위가 밝은 바깥), -1: 아래 경계. 바깥(밝음) 20~60px, 안쪽(어두움) 10~30px 중앙값
    out = c[e - 60:e - 20] if sgn > 0 else c[e + 21:e + 61]
    ins = c[e + 10:e + 30] if sgn > 0 else c[e - 29:e - 9]
    hi, lo = np.median(out), np.median(ins)
    seg = c[e - 20:e + 21] if sgn > 0 else c[e + 20:e - 21:-1]   # 바깥 -> 안쪽 방향으로 정렬
    f = (seg - lo) / (hi - lo)                                    # 1 = 밝음, 0 = 어두움
    def cross(level):
        s = np.where((f[:-1] >= level) & (f[1:] < level))[0]
        if len(s) == 0: return np.nan
        i = s[0]; return i + (f[i] - level) / (f[i] - f[i + 1])
    return cross(0.1) - cross(0.9), hi, lo
fs = sorted([f for f in glob.glob(SHARE + '[12]_*/NO*/*.jpg') if '_1533' not in f], key=os.path.basename)
rows = []
for f in fs:
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    r = dict(file=os.path.basename(f), stage=f.split('/')[-3][0], no=f.split('/')[-2])
    for name, xs in OBJ.items():
        acc = {'wt': [], 'wb': [], 'hi': [], 'lo': []}
        for x in xs:
            c = g[Y0:Y1, x - 1:x + 2].mean(1)
            dk = np.r_[0, (c < 100).astype(int), 0]
            st, en = np.where(np.diff(dk) == 1)[0], np.where(np.diff(dk) == -1)[0] - 1
            k = (st > 0) & (en < len(c) - 1) & (en - st >= 10); st, en = st[k], en[k]
            if len(st) == 0: continue
            wt, h1, l1 = prof(c, st[0], +1); wb, h2, l2 = prof(c, en[-1], -1)
            acc['wt'].append(wt); acc['wb'].append(wb); acc['hi'] += [h1, h2]; acc['lo'] += [l1, l2]
        for k2, v in acc.items(): r[f'{name}_{k2}'] = np.nanmedian(v)
    rows.append(r); print(r['file'], f"{r['pin_wt']:.2f} {r['pin_wb']:.2f}", flush=True)
pd.DataFrame(rows).to_csv('edge_width.csv', index=False); print('DONE')
