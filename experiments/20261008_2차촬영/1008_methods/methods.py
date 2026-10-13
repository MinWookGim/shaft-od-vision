# 10-08 130장: 경계 찾는 방법 후보를 같은 평가 규칙으로 비교 (10-11, 결과 보기 전에 규칙 고정)
# 평가 규칙
#   - 사진마다 지름 px / 핀 px x 10 = mm, 시편(묶음)마다 10장 평균
#   - 하나씩 빼고 교정: 정상 시편 하나를 빼고 나머지 정상 시편으로 치우침을 구해 뺀 오차. 모든 정상 시편에 반복
#   - 정상 시편 = 묶음 2~10, 13 (묶음 1 = NO1 첫 촬영은 핀을 다시 놓기 전이라 제외. 이 규칙만 미리 정함)
#   - NG 두 개는 정상 10개 전부로 교정한 뒤 오차
#   - "평균 찍기" = 측정 없이 모든 시편을 정상 평균으로 찍었을 때 오차. 이것보다 나아야 측정한 의미가 있다
import os, csv, numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__))
C = np.load(os.path.join(HERE, '_out', 'cache.npz'), allow_pickle=True)
DATA = C['data'][0]; META = C['meta'][0]
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
SY0 = META['SY'][0]
NORMAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13]; NG = [11, 12]

# ---------------- 한 열에서 경계 찾기 ----------------
def cross(col, thr, i, step):
    """i 에서 step 방향으로 가며 col 이 thr 아래로 처음 내려가는 곳(소수점 보간). step=+1 위->아래, -1 아래->위"""
    n = len(col)
    while 0 <= i < n and col[i] >= thr: i += step
    if not (0 < i < n - 1): return np.nan
    a, b = col[i - step], col[i]                   # a 는 밝은 쪽, b 는 어두운 쪽
    return (i - step) + step * (a - thr) / (a - b)

def m0_half(col, bg, lo, hi):
    """내 기존 방법: 배경과 어두운 쪽(5% 백분위)의 중간값을 바깥에서 안쪽으로 처음 지나는 곳"""
    seg = col[lo:hi]; thr = (bg + np.percentile(seg, 5)) / 2
    return lo + cross(seg, thr, 0, +1), lo + cross(seg, thr, len(seg) - 1, -1)

def m2_outer(col, bg, lo, hi, win=15):
    """바깥 단 고정: 배경의 90% 아래로 처음 떨어지는 곳(바깥에서 안쪽) 근처 win px 안의 가장 어두운 값을
    그 단의 바닥으로 보고, 배경과 그 바닥의 중간값을 지나는 곳. 단이 여러 개여도 가장 바깥 단만 본다"""
    seg = col[lo:hi]; n = len(seg)
    out = []
    for start, step in ((0, +1), (n - 1, -1)):
        j = start
        while 0 <= j < n and seg[j] >= 0.9 * bg: j += step
        if not (0 <= j < n): return np.nan, np.nan
        k = j + step * win
        inner = seg[min(j, k):max(j, k) + 1]
        thr = (bg + inner.min()) / 2
        out.append(lo + cross(seg, thr, start, step))
    return out[0], out[1]

def m3_centroid(col, bg, lo, hi, win=15):
    """바깥 단의 밝기 변화율 무게중심: 단 하나의 '가운데'를 변화율 가중평균으로"""
    t, b = m2_outer(col, bg, lo, hi, win)
    if np.isnan(t) or np.isnan(b): return np.nan, np.nan
    g = np.abs(np.gradient(col.astype(float)))
    res = []
    for e in (t, b):
        i0, i1 = int(e) - 6, int(e) + 7
        w = g[i0:i1]; y = np.arange(i0, i1)
        res.append((w * y).sum() / w.sum())
    return res[0], res[1]

def m5_erf(col, bg, lo, hi, win=15):
    """경계 모양 맞추기: 바깥 단 주변 +-12px 에 흐린 계단(erf) 모양을 맞춰 계단 가운데를 찾는다"""
    t, b = m2_outer(col, bg, lo, hi, win)
    if np.isnan(t) or np.isnan(b): return np.nan, np.nan
    res = []
    for e, sgn in ((t, -1), (b, +1)):
        i0, i1 = int(e) - 12, int(e) + 13
        y = np.arange(i0, i1, dtype=float); v = col[i0:i1].astype(float)
        f = lambda y, y0, s, a, c: a + c * 0.5 * (1 + erf(sgn * (y - y0) / (np.sqrt(2) * abs(s) + 1e-6)))
        try:
            p, _ = curve_fit(f, y, v, p0=[e, 2.0, v.min(), v.max() - v.min()], maxfev=2000)
            res.append(p[0] if abs(p[0] - e) < 8 else np.nan)
        except Exception:
            res.append(np.nan)
    return res[0], res[1]

FINDERS = {'M0_50%경계(기존)': m0_half, 'M2_바깥단50%': m2_outer, 'M3_바깥단_변화율중심': m3_centroid, 'M5_바깥단_계단모양맞춤': m5_erf}

# ---------------- 사진 한 장 -> 지름 px, 핀 px ----------------
S_LO, S_HI = 3600 - SY0, 4320 - SY0     # 샤프트 찾는 범위 (위 가림막 경계 3465 는 안 들어옴)
P_LO, P_HI = 0, 400                       # 핀 범위 전체
def measure(rec, finder, line_fit=False):
    sh, pin, bg = rec['shaft'].astype(float), rec['pin'].astype(float), rec['bg']
    xs = np.arange(0, sh.shape[1], 2)
    tb = np.array([finder(sh[:, x], bg, S_LO, S_HI) for x in xs])
    ok = ~np.isnan(tb).any(1)
    if line_fit:
        # 2D 직선 맞추기: 위·아래 경계 각각에 직선을 맞추고(크게 벗어난 점 2번 걸러냄), 가운데 x 에서 거리 x cos(기울기)
        xc = xs[ok].mean(); vals = []
        for k in (0, 1):
            x, y = xs[ok], tb[ok, k]
            for _ in range(2):
                p = np.polyfit(x, y, 1); r = y - np.polyval(p, x); m = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r)) + 1e-9
                x, y = x[m], y[m]
            vals.append(p)
        slope = (vals[0][0] + vals[1][0]) / 2
        d = (np.polyval(vals[1], xc) - np.polyval(vals[0], xc)) * np.cos(np.arctan(slope))
    else:
        d = np.median(tb[ok, 1] - tb[ok, 0])
    pb = np.array([finder(pin[:, x], bg, P_LO, P_HI) for x in range(35, 80, 2)])   # 핀 열 9655~9700 (기존과 같은 자리)
    pok = ~np.isnan(pb).any(1)
    return d, np.median(pb[pok, 1] - pb[pok, 0]), np.median(tb[ok, 0]) + SY0

def top_only(rec):
    """위 경계만: 위 가림막 경계(고정물)에서 샤프트 위 경계까지 거리. 지름이 크면 윗면이 올라가 거리가 줄어든다"""
    sh, bg = rec['shaft'].astype(float), rec['bg']
    hs = []
    for x in range(0, sh.shape[1], 2):
        col = sh[:, x]
        thr = (bg + col[:40].mean()) / 2                 # 가림막 경계: 맨 위 어두움 -> 아래로 가며 밝아짐
        i = int(np.argmax(col[:S_LO] > thr))             # 처음 밝아지는 곳
        s = (i - 1) + (thr - col[i - 1]) / (col[i] - col[i - 1]) if i > 0 else np.nan
        t, _ = m0_half(col, bg, S_LO, S_HI)
        hs.append(t - s)
    return np.nanmedian(hs)

# ---------------- 평가 ----------------
def evaluate(est, label):
    """est: 묶음 -> 사진별 mm 배열. 하나씩 빼고 교정 오차"""
    true = {int(r['묶음']): float(r['실측mm']) for r in rows}
    mean = {g: np.mean(v) for g, v in est.items()}
    rep = np.median([np.std(v, ddof=1) for v in est.values()]) * 1000
    loo = {}
    for g in NORMAL:
        others = [h for h in NORMAL if h != g]
        off = np.mean([mean[h] - true[h] for h in others])
        loo[g] = (mean[g] - off - true[g]) * 1000
    off_all = np.mean([mean[h] - true[h] for h in NORMAL])
    ng = {g: (mean[g] - off_all - true[g]) * 1000 for g in NG}
    e = np.array(list(loo.values()))
    return dict(방법=label, 흔들림um=round(rep, 1), LOO_MAE=round(np.abs(e).mean(), 1), LOO_최대=round(np.abs(e).max(), 1),
                NG1=round(ng[11], 1), NG2=round(ng[12], 1), 시편별=' '.join(f'{loo[g]:+.0f}' for g in NORMAL))

def baseline_mean():
    true = {int(r['묶음']): float(r['실측mm']) for r in rows}
    e = [(np.mean([true[h] for h in NORMAL if h != g]) - true[g]) * 1000 for g in NORMAL]
    ng = [(np.mean([true[h] for h in NORMAL]) - true[g]) * 1000 for g in NG]
    return dict(방법='(기준) 평균 찍기 - 측정 안 함', 흔들림um=0, LOO_MAE=round(np.abs(e).mean(), 1), LOO_최대=round(np.abs(np.array(e)).max(), 1),
                NG1=round(ng[0], 1), NG2=round(ng[1], 1), 시편별=' '.join(f'{v:+.0f}' for v in e))

if __name__ == '__main__':
    res = [baseline_mean()]
    per_img = []
    variants = [(n, f, False) for n, f in FINDERS.items()] + [('M4_바깥단50%+2D직선', m2_outer, True)]
    for name, f, lf in variants:
        est = {}
        for r in rows:
            d, p, top = measure(DATA[r['파일']], f, lf)
            est.setdefault(int(r['묶음']), []).append(d / p * 10)
            per_img.append([r['파일'], r['묶음'], name, d, p, top])
        res.append(evaluate(est, name)); print(res[-1], flush=True)
    # 위 경계만: 거리 h(px) 를 핀 px 로 mm 환산. 90도 V블록이면 지름 변화 = -h 변화 / 1.207
    est = {}
    pin0 = {r['파일']: measure(DATA[r['파일']], m0_half)[1] for r in rows}
    for r in rows:
        h = top_only(DATA[r['파일']]) / pin0[r['파일']] * 10
        est.setdefault(int(r['묶음']), []).append(20.0 - h / 1.207)       # 20.0 은 아무 상수(교정이 치우침으로 흡수)
    res.append(evaluate(est, 'M1_위경계만(V블록 원리)')); print(res[-1])
    w = csv.DictWriter(open(os.path.join(HERE, '_out', 'methods_result.csv'), 'w', newline='', encoding='utf-8-sig'), fieldnames=list(res[0]))
    w.writeheader(); w.writerows(res)
    csv.writer(open(os.path.join(HERE, '_out', 'methods_per_image.csv'), 'w', newline='', encoding='utf-8-sig')).writerows(
        [['파일', '묶음', '방법', '지름px', '핀px', '위경계y']] + per_img)
