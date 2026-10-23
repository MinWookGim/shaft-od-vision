# 4번: 암막 없음 세트(앞에서 빛 받음)에서 핀·샤프트 지름을 다시 측정
# 기존 50% 방식은 창 양끝 10% 평균을 '바깥/안쪽 밝기'로 쓰는데, 앞빛이면 몸통 안에 255 반사띠가 있어 안쪽 기준이 틀어짐
# -> 국소 방식: 경계 근처만 보고, 바깥 밝기 = 경계 바깥 20~60px 중앙값, 안쪽 밝기 = 경계 안쪽 10~30px 중앙값, 그 중간값을 지나는 점(선형보간)
# ★ 하다 보니 기존 50% 방식(run_v3_d20.py 의 tb)의 결함 발견: 암막 사진은 창 맨 위(y 3000~3160)가 어두운 띠라
#   위 경계의 '바깥 밝기'가 배경(약 180)이 아니라 어두운 띠(약 14)로 잡혀 위 경계를 바닥(밝기 약 11)에서 잡고 있었음
#   -> 사용 190장 전부를 이 국소 방식으로 다시 측정 (local_edges.csv)
# 출력: local_edges.csv
import csv
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SHARE = Path('../../field/photos/20261021_원전기계/3차_261021_분류')
NO1_OFF = (2408, 1112); Y0, Y1 = 3150, 4350
PIN_X = range(5300, 5800, 10); SH_X = range(8015, 8335, 10)

def refine(c, e, sgn):
    # sgn=+1: 위 경계(위가 바깥), -1: 아래 경계(아래가 바깥). e = 대략 위치
    out = c[e - 60:e - 20] if sgn > 0 else c[e + 21:e + 61]
    ins = c[e + 10:e + 30] if sgn > 0 else c[e - 29:e - 9]
    h = (np.median(out) + np.median(ins)) / 2
    lo, hi = e - 20, e + 20
    seg = c[lo:hi]
    s = np.where(np.diff(np.sign(seg - h)) != 0)[0]
    if len(s) == 0: return np.nan
    i = s[0] if sgn > 0 else s[-1]
    return lo + i + (h - seg[i]) / (seg[i + 1] - seg[i])

def height(g, x, dy):
    c = g[Y0 - dy:Y1 - dy, x]
    # 어두운(100 미만) 연속 구간들 중 창 끝에 닿은 것은 뺀다(암막 사진은 창 맨 위가 원래 어두운 띠)
    # 남은 것 중 첫 구간 시작 = 위 경계, 마지막 구간 끝 = 아래 경계 (앞빛 사진은 몸통 안 반사띠가 구간을 쪼개므로 '가장 긴 구간'을 쓰면 안 됨)
    dk = np.r_[0, (c < 100).astype(int), 0]
    st, en = np.where(np.diff(dk) == 1)[0], np.where(np.diff(dk) == -1)[0] - 1
    keep = (st > 0) & (en < len(c) - 1) & (en - st >= 10)   # 배경 티끌(1~3px 어두운 점)은 빼기 (10-23 새벽 수정)
    st, en = st[keep], en[keep]
    if len(st) == 0: return np.nan, np.nan, np.nan
    t = refine(c, st[0], +1); b = refine(c, en[-1], -1)
    return b - t, t, b

def meas(g, xs, dx, dy):
    r = np.array([height(g, x - dx, dy) for x in xs])
    return np.nanmedian(r[:, 0]), np.nanstd(r[:, 0]), np.nanmedian(r[:, 1]) + Y0, np.nanmedian(r[:, 2]) + Y0

ps = sorted(SHARE.glob('*/NO*/*.jpg'), key=lambda q: q.name)
w = csv.writer(open('local_edges.csv', 'w', newline=''))
w.writerow(['file', 'stage', 'no', 'pin', 'pin_sd', 'pin_top', 'pin_bot', 'shaft', 'shaft_sd', 'sh_top', 'sh_bot'])
for p in ps:
    dx, dy = NO1_OFF if '_1533' in p.name else (0, 0)
    g = np.asarray(Image.open(p).convert('L'), dtype=float)
    w.writerow([p.name, p.parts[-3][0], p.parts[-2]] + [f'{v:.3f}' for v in meas(g, PIN_X, dx, dy) + meas(g, SH_X, dx, dy)])
    print(p.name, flush=True)
