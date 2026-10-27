# 10-27 멘토 제안: 대표값 후보(Median, Mean, P90, P95, 상위3·5 평균, Max, 경계 envelope) 중 무엇이 실측과 가장 가까운가
# 1단계: 3차 본시험(암막) 120장 + 재장착 30장에서 Ø20 단 31개 열의 위·아래 경계(국소 50%, 열마다 3px 평균 단면)를 저장
# 경계 측정 방식은 1021_main/local_edges.py 와 같다(창 끝에 닿은 어두운 띠 제외, 경계 바깥 20~60 / 안쪽 10~30px 중간값)
import csv, glob, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
NO1_OFF = (2408, 1112); Y0, Y1 = 3150, 4350
XS = np.linspace(8030, 8320, 31).round().astype(int)   # Ø20 단 평평한 구간(약 8000~8350)의 안쪽 31개 열

def refine(c, e, sgn):
    out = c[e - 60:e - 20] if sgn > 0 else c[e + 21:e + 61]
    ins = c[e + 10:e + 30] if sgn > 0 else c[e - 29:e - 9]
    h = (np.median(out) + np.median(ins)) / 2
    lo = e - 20; seg = c[lo:e + 20]
    s = np.where(np.diff(np.sign(seg - h)) != 0)[0]
    if len(s) == 0: return np.nan
    i = s[0] if sgn > 0 else s[-1]
    return lo + i + (h - seg[i]) / (seg[i + 1] - seg[i])

def edges(g, x, dx, dy):
    c = g[Y0 - dy:Y1 - dy, x - dx - 1:x - dx + 2].mean(1)
    dk = np.r_[0, (c < 100).astype(int), 0]
    st, en = np.where(np.diff(dk) == 1)[0], np.where(np.diff(dk) == -1)[0] - 1
    keep = (st > 0) & (en < len(c) - 1) & (en - st >= 10)
    st, en = st[keep], en[keep]
    if len(st) == 0: return np.nan, np.nan
    return refine(c, st[0], +1), refine(c, en[-1], -1)

fs = sorted(glob.glob(SHARE + '1_*/NO*/*.jpg') + glob.glob(SHARE + '2_*/NO*/*.jpg'), key=os.path.basename)
w = csv.writer(open('repr_cols.csv', 'w', newline=''))
w.writerow(['file', 'stage', 'no'] + [f't{i}' for i in range(31)] + [f'b{i}' for i in range(31)])
for f in fs:
    n = os.path.basename(f); dx, dy = NO1_OFF if '_1533' in n else (0, 0)
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    r = [edges(g, x, dx, dy) for x in XS]
    w.writerow([n, f.split('/')[-3][0], f.split('/')[-2]] + [f'{t:.3f}' for t, b in r] + [f'{b:.3f}' for t, b in r])
    print(n, flush=True)
print('DONE')
