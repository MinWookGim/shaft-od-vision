# OpenCV(경계 읽는 규칙)를 고쳐서 맞출 수 있나 -> "정답을 보고 규칙을 골라도" 사진이 지름을 담고 있는지 본다 (10-12)
# 규칙 가족: 위 경계 문턱 tu, 아래 경계 문턱 tl (어두운 바닥~배경 사이 10%~90%, 바깥에서 안쪽으로 처음 지나는 곳)
#   형태학 전처리(OPEN/CLOSE/TOPHAT 등)도 결국 경계가 어느 높이에 걸리느냐를 옮기는 일이라 이 가족이 대부분 덮는다
# 판정 1: 81개 규칙 중 가장 좋은 것의 상관 r(시편 12개 평균 vs 실측) -> 실측을 섞어 같은 짓을 한 것(우연)과 비교
# 판정 2: 정답 보고 고른 최선 규칙의 LOO MAE(정상 10) -> 평균 찍기 5.8 와 비교 (이건 낙관적 상한, 실제로는 더 나쁘다)
import os, sys, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, '..', '1008_methods'); sys.path.insert(0, M)
import methods as MT                                    # 캐시·행·핀 측정(M0)·평가 규칙을 그대로 쓴다

TS = np.round(np.arange(0.1, 0.91, 0.1), 1)              # 문턱 후보 9개
true = {int(r['묶음']): float(r['실측mm']) for r in MT.rows}
BUN = MT.NORMAL + MT.NG                                  # 12묶음 (묶음1 = NO1 첫 촬영은 기존 규칙대로 제외)

def first_cross(n, t, lo, hi, down):
    """n: 정규화 밝기(0=바닥,1=배경) 행렬[H,W]. lo~hi 안에서 바깥->안쪽으로 처음 t 아래로 내려가는 소수점 행"""
    seg = n[lo:hi] if down else n[lo:hi][::-1]
    below = seg < t
    i = np.argmax(below, axis=0)                         # 처음 내려간 칸
    ok = below.any(0) & (i > 0)
    i = np.clip(i, 1, None)
    cols = np.arange(seg.shape[1])
    a, b = seg[i - 1, cols], seg[i, cols]                # a 밝은 쪽, b 어두운 쪽
    y = (i - 1) + (a - t) / (a - b + 1e-9)
    y = np.where(ok, y, np.nan)
    return lo + y if down else hi - 1 - y

def diam_table(rec):
    """한 장 -> [tu, tl] 81칸 지름 px (열 중앙값)"""
    sh = rec['shaft'].astype(float)[:, ::2]
    seg = sh[MT.S_LO:MT.S_HI]
    floor = np.percentile(seg, 5, axis=0); bg = rec['bg']
    n = (sh - floor) / (bg - floor)
    tops = {t: first_cross(n, t, MT.S_LO, MT.S_HI, True) for t in TS}
    bots = {t: first_cross(n, t, MT.S_LO, MT.S_HI, False) for t in TS}
    return np.array([[np.nanmedian(bots[tl] - tops[tu]) for tl in TS] for tu in TS])

if __name__ == '__main__':
    per = {}                                             # 묶음 -> 사진별 [9,9] mm
    for r in MT.rows:
        g = int(r['묶음'])
        if g not in BUN: continue
        rec = MT.DATA[r['파일']]
        _, pin, _ = MT.measure(rec, MT.m0_half)          # 핀은 깨끗한 경계라 기존 M0 그대로
        per.setdefault(g, []).append(diam_table(rec) / pin * 10)
    mean = np.array([np.mean(per[g], 0) for g in BUN])   # [12,9,9]
    sd = np.array([np.std(per[g], 0, ddof=1) for g in BUN]) * 1000
    y = np.array([true[g] for g in BUN])

    def rmap(yv):
        X = mean.reshape(12, -1); Xc = X - X.mean(0); yc = yv - yv.mean()
        return (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    r = rmap(y)
    best = int(np.nanargmax(r)); tu, tl = TS[best // 9], TS[best % 9]
    rng = np.random.default_rng(0)
    null = np.array([np.nanmax(rmap(rng.permutation(y))) for _ in range(5000)])
    p = (null >= r[best]).mean()

    # 같은 규칙의 LOO (기존 평가 규칙: 핀 환산 mm, 치우침만 빼기)
    def loo(k):
        est = {g: [a.reshape(-1)[k] for a in per[g]] for g in BUN}
        return MT.evaluate(est, f'tu{TS[k//9]}_tl{TS[k%9]}')
    allres = [loo(k) for k in range(81)]
    maes = np.array([a['LOO_MAE'] for a in allres])
    out = []
    out.append(f'규칙 81개(위 문턱 x 아래 문턱), 시편 12개(정상10+NG2), 실측 sd {np.std(y,ddof=1)*1000:.1f}um')
    out.append(f'같은 사진 10장 sd 중앙값: {np.median(sd):.1f}um  -> 10장 평균의 흔들림 약 {np.median(sd)/np.sqrt(10):.1f}um')
    out.append(f'[판정1] 정답 보고 고른 최선 상관 r={r[best]:+.2f} (tu={tu}, tl={tl}). 실측을 섞은 5000번의 최선 r: 중앙 {np.median(null):+.2f}, 95% {np.percentile(null,95):+.2f}  -> p={p:.3f}')
    out.append(f'  상관 r 분포(81개): 최소 {np.nanmin(r):+.2f}, 중앙 {np.nanmedian(r):+.2f}, 최대 {np.nanmax(r):+.2f}')
    kb = int(np.argmin(maes))
    out.append(f'[판정2] 정답 보고 고른 최선 LOO MAE {maes[kb]:.1f}um ({allres[kb]["방법"]}, 최대 {allres[kb]["LOO_최대"]}, NG1 {allres[kb]["NG1"]:+}, NG2 {allres[kb]["NG2"]:+}) vs 평균 찍기 5.8')
    out.append(f'  81개 MAE 분포: 최소 {maes.min():.1f}, 중앙 {np.median(maes):.1f}, 최대 {maes.max():.1f}')
    out.append(f'  최선 상관 규칙의 LOO: {allres[best]}')
    out.append('위 문턱(행) x 아래 문턱(열) 상관 r:')
    out.append('      ' + ' '.join(f'{t:>5}' for t in TS))
    for i, t in enumerate(TS):
        out.append(f'{t:>5} ' + ' '.join(f'{v:+.2f}' for v in r[i*9:(i+1)*9]))
    out.append('위 문턱(행) x 아래 문턱(열) LOO MAE um:')
    for i, t in enumerate(TS):
        out.append(f'{t:>5} ' + ' '.join(f'{v:5.1f}' for v in maes[i*9:(i+1)*9]))
    txt = '\n'.join(out); print(txt)
    open(os.path.join(HERE, 'result.txt'), 'w').write(txt + '\n')
