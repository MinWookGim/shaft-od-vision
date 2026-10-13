# 핀 열 위치 효과 분석 (규칙은 돌리기 전에 고정)
#   xL = 핀 왼쪽 끝 (폭이 250px 넘는 첫 열), xR = 측정 가능한 마지막 열(V블록 끝 그림자 직전)
#   s = x - xL (핀 위 위치, 왼쪽 끝에서 px)
#   핀 px 후보
#     A 고정 열 9655~9700 (10-08 분석 그대로)
#     B 핀 위 같은 자리 s 60~105 (핀 기준으로 A 와 같은 길이 45px, 핀이 P2 일 때 A 와 거의 같은 자리)
#     C 핀 위 왼쪽 s 25~70 (그림자에서 가장 먼 쪽)
#   평가: 묶음별 핀 평균의 퍼짐(초점 구간 안), 지름 LOO MAE (샤프트는 M0 그대로, methods.py 와 같은 규칙), NG 오차
import os, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, '_out', 'pin_scan.npz')); XS, TOP, BOT, FILES = z['XS'], z['TOP'], z['BOT'], list(z['files'])
W = BOT - TOP
rows = {r['파일']: r for r in csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig'))}
sh = {r['파일']: float(r['지름px']) for r in csv.DictReader(open(os.path.join(HERE, '..', '1008_methods', '_out', 'methods_per_image.csv'), encoding='utf-8-sig'))
      if r['방법'] == 'M0_50%경계(기존)'}
NORMAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13]; NG = [11, 12]
out = []

def ends(w):
    ok = np.where((w > 250) & (XS >= 9500))[0]          # 9450 근처 다른 어두운 물체 제외
    # 핀 구간 = 왼쪽에서 처음 250 넘은 곳부터 연속된 덩어리
    i0 = ok[0]; i1 = i0
    while i1 + 1 < len(w) and not np.isnan(w[i1 + 1]) and w[i1 + 1] > 250: i1 += 1
    return i0, i1

info = []
for k, f in enumerate(FILES):
    i0, i1 = ends(W[k]); xL, xR = XS[i0], XS[i1]
    s = XS - xL
    def band(lo, hi, by_s):
        m = ((s >= lo) & (s <= hi)) if by_s else ((XS >= lo) & (XS <= hi))
        return np.nanmedian(W[k, m])
    info.append(dict(f=f, g=int(rows[f]['묶음']), focus=rows[f]['초점'], xL=xL, xR=xR,
                     A=band(9655, 9700, False), B=band(60, 105, True), C=band(25, 70, True),
                     top_slope=np.polyfit(s[(s >= 20) & (s <= xR - xL)], TOP[k, (s >= 20) & (s <= xR - xL)], 1)[0],
                     bot_slope=np.polyfit(s[(s >= 20) & (s <= xR - xL)], BOT[k, (s >= 20) & (s <= xR - xL)], 1)[0]))
true = {int(r['묶음']): float(r['실측mm']) for r in rows.values()}

def P(*a):
    line = ' '.join(str(x) for x in a); print(line); out.append(line)

P('== 핀 자리 (묶음별 중앙값) : 왼쪽 끝 xL, 측정 가능 끝 xR, 핀 길이 xR-xL, 위 경계 기울기(px/100px), 아래 경계 기울기')
for g in range(1, 14):
    d = [i for i in info if i['g'] == g]
    P(f"묶음{g:2d} {d[0]['focus']} xL {np.median([i['xL'] for i in d]):.0f} xR {np.median([i['xR'] for i in d]):.0f} "
      f"길이 {np.median([i['xR'] - i['xL'] for i in d]):.0f}  위기울기 {np.median([i['top_slope'] for i in d]) * 100:+.2f} 아래기울기 {np.median([i['bot_slope'] for i in d]) * 100:+.2f}  "
      f"핀A {np.mean([i['A'] for i in d]):.2f} B {np.mean([i['B'] for i in d]):.2f} C {np.mean([i['C'] for i in d]):.2f}")

P('\n== 핀 위치(s)별 폭: 전체 130장 평균 프로파일 (각 사진 자기 중앙값을 뺀 값)')
prof = {}
for k, i in enumerate(info):
    s = XS - i['xL']; m = (s >= 0) & (s <= i['xR'] - i['xL'])
    ref = np.nanmedian(W[k, m & (s >= 20)])
    for ss, w in zip(s[m], W[k, m]): prof.setdefault(int(ss // 20 * 20), []).append(w - ref)
for ss in sorted(prof):
    if len(prof[ss]) > 30: P(f's {ss:3d}~{ss + 19:3d}: {np.mean(prof[ss]):+.2f}px (n={len(prof[ss])})')

P('\n== 평가 (지름 = 샤프트 M0 px / 핀 px x 10)')
def evaluate(key):
    est = {}
    for i in info: est.setdefault(i['g'], []).append(sh[i['f']] / i[key] * 10)
    mean = {g: np.mean(v) for g, v in est.items()}
    loo = []
    for g in NORMAL:
        off = np.mean([mean[h] - true[h] for h in NORMAL if h != g]); loo.append((mean[g] - off - true[g]) * 1000)
    off = np.mean([mean[h] - true[h] for h in NORMAL]); ng = [(mean[g] - off - true[g]) * 1000 for g in NG]
    pin = {g: np.mean([i[key] for i in info if i['g'] == g]) for g in range(1, 14)}
    spread = {f: np.ptp([pin[g] for g in pin if info[[i['g'] for i in info].index(g)]['focus'] == f]) for f in ('F1', 'F2', 'F3')}
    rep = np.median([np.std([i[key] for i in info if i['g'] == g], ddof=1) for g in range(1, 14)])
    P(f'{key}: LOO MAE {np.mean(np.abs(loo)):5.1f} 최대 {np.max(np.abs(loo)):5.1f}um | NG {ng[0]:+.0f}, {ng[1]:+.0f} | '
      f'초점구간 안 묶음간 핀 폭 범위 F1 {spread["F1"]:.2f} F2 {spread["F2"]:.2f} F3 {spread["F3"]:.2f}px | 같은 묶음 핀 sd {rep:.2f}px')
    P('   시편별 ' + ' '.join(f'{v:+.0f}' for v in loo))
for key in ('A', 'B', 'C'): evaluate(key)
P('(기준) 평균 찍기 LOO MAE 5.8 최대 12.7um')
open(os.path.join(HERE, '_out', 'analyze.txt'), 'w').write('\n'.join(out))
