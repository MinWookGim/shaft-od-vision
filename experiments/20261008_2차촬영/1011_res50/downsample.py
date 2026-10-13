# 2억 화소(10-08) 사진을 줄여서 50MP(가로세로 1/2)·12.5MP(1/4)처럼 만들고 같은 측정을 다시 돌린다
# 줄이기 = 화소 k x k 묶음 평균 (50MP 는 실제로 화소 4개를 묶어 읽는다)
# 측정 = 1008_methods 의 M0(50% 경계) 와 M2(바깥 단 50%), 핀 열 9655~9700, 평가도 같은 함수(evaluate)
# 규칙은 돌리기 전에 고정: 지표 = LOO MAE/최대, 같은 시편 10장 흔들림, F3 안정 구간(NO8,9,10,NO1재) sd
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '1008_methods'))
import methods as M                                  # cache.npz 를 읽고 경계 함수·평가 함수를 그대로 쓴다

def binning(a, k):
    # k x k 화소를 평균해 한 화소로 (끝에 남는 줄은 버림)
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].astype(float).reshape(h // k, k, w // k, k).mean((1, 3))

def measure_k(rec, finder, k):
    sh, pin, bg = binning(rec['shaft'], k), binning(rec['pin'], k), rec['bg']
    lo, hi = M.S_LO // k, M.S_HI // k                      # 샤프트 찾는 범위도 같은 비율로
    tb = np.array([finder(sh[:, x], bg, lo, hi) for x in range(sh.shape[1])])
    ok = ~np.isnan(tb).any(1)
    d = np.median(tb[ok, 1] - tb[ok, 0])
    pb = np.array([finder(pin[:, x], bg, 0, pin.shape[0]) for x in range(35 // k, 80 // k)])   # 핀 열 9655~9700
    pok = ~np.isnan(pb).any(1)
    return d, np.median(pb[pok, 1] - pb[pok, 0])

def m2_k(k):
    # 바깥 단 창(win 15px)도 같은 실제 길이가 되게 줄인다
    return lambda col, bg, lo, hi: M.m2_outer(col, bg, lo, hi, win=max(3, 15 // k))

if __name__ == '__main__':
    true = {int(r['묶음']): float(r['실측mm']) for r in M.rows}
    out = []
    def P(s): print(s, flush=True); out.append(s)
    P(str(M.baseline_mean()))
    for name, mk in (('M0', lambda k: M.m0_half), ('M2', m2_k)):
        for k in (1, 2, 4):
            est = {}; px = {}
            for r in M.rows:
                d, p = measure_k(M.DATA[r['파일']], mk(k), k)
                est.setdefault(int(r['묶음']), []).append(d / p * 10); px.setdefault(int(r['묶음']), []).append(p)
            res = M.evaluate(est, f'{name} 1/{k} ({200 // k // k}MP 상당, 1px={37 * k}um)')
            # F3 안정 구간: 정상 전체로 교정한 뒤 NO8,NO9,NO10,NO1재(묶음 8,9,10,13) 오차
            off = np.mean([np.mean(est[h]) - true[h] for h in M.NORMAL])
            f3 = [(np.mean(est[g]) - off - true[g]) * 1000 for g in (8, 9, 10, 13)]
            pin_sd = np.median([np.std(v, ddof=1) for v in px.values()])
            P(f"{res['방법']}: LOO MAE {res['LOO_MAE']} 최대 {res['LOO_최대']} | 10장 흔들림 {res['흔들림um']}um | 핀px 10장 sd {pin_sd:.3f}(x{k}={pin_sd * k:.3f} 원본px) | "
              f"NG {res['NG1']:+.0f},{res['NG2']:+.0f} | F3안정 {' '.join(f'{v:+.0f}' for v in f3)} sd {np.std(f3, ddof=1):.1f}")
            P(f"   시편별 {res['시편별']}")
    open(os.path.join(HERE, 'result.txt'), 'w').write('\n'.join(out))
