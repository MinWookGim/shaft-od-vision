# 노이즈 바닥: 같은 1/2 줄이기인데 묶음 격자를 1px 옮기면(위상) 시편별 오차가 얼마나 바뀌나
# 옮겨도 정보량은 같다. 이 차이만큼은 해상도 탓이 아니라 우연이다
import numpy as np, downsample as D
M = D.M
for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
    est = {}
    for r in M.rows:
        rec = dict(M.DATA[r['파일']]); rec['shaft'] = rec['shaft'][dy:, dx:]; rec['pin'] = rec['pin'][dy:, dx:]
        d, p = D.measure_k(rec, M.m0_half, 2)
        est.setdefault(int(r['묶음']), []).append(d / p * 10)
    res = M.evaluate(est, f'M0 1/2 위상({dy},{dx})')
    print(res['방법'], 'LOO MAE', res['LOO_MAE'], '최대', res['LOO_최대'], '| 시편별', res['시편별'], flush=True)
