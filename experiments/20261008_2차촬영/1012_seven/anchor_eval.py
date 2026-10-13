# 선행 개발자 박스(고정 / Anchor 추적)로 seven.py 와 같은 표를 만든다
# 박스 높이 1000px(샤프트 약 544px) 이라 허용 지름범위를 박스 높이의 0.40~0.70 으로 둔다(F 의 0.55~0.95 는 720px 박스 기준이었음)
import os, numpy as np, seven as S
import shaft_core as sc
PIN_P = S.P                                       # 핀은 seven.py 와 같은 설정(핀 박스 330px 기준)
SHAFT_P = sc.Params(vertical_margin=0.0, dia_basis='box', edge_select='strongest', dia_min_box=0.40, dia_max_box=0.70)
A = np.load(os.path.join(S.HERE, 'anchor_crops.npz'), allow_pickle=True)['data'][0]
out = []
def Pr(s): print(s, flush=True); out.append(s)
sc_by = {}
for r in S.rows: sc_by.setdefault(r['시편'], []).append(A[r['파일']]['score'])
Pr('매칭 점수(시편별 중앙값): ' + ' '.join(f'{k} {np.median(v):.3f}' for k, v in sc_by.items()))
Pr('평균 찍기: 선행 나누기 MAE 3.5um / 정상10 LOO 5.8um')
for boxname in ('fixed', 'track'):
    for vname in ('0 원본(F)', '2 OPEN+CLOSE', '7 중앙 mask', '1 inRange', '전부'):
        f = S.VARS[vname]; sh, pn = {}, {}
        for r in S.rows:
            g = int(r['묶음'])
            S.P = SHAFT_P; sh.setdefault(g, []).append(S.meas(f(A[r['파일']][boxname])))
            S.P = PIN_P; pn.setdefault(g, []).append(S.meas(f(S.D[r['파일']]['pin'])))
        m = {g: np.nanmean(v) for g, v in sh.items()}
        a, b = np.polyfit([m[g] for g in S.CAL], [S.true[g] for g in S.CAL], 1)
        mae = np.mean([abs(a * m[g] + b - S.true[g]) for g in S.VAL]) * 1000
        vok = sum(S.ok(a * x + b) == 'OK' for g in S.VAL for x in sh[g] if not np.isnan(x))
        ngok = sum(S.ok(a * x + b) == 'OK' for g in S.NG for x in sh[g] if not np.isnan(x))
        est = {g: np.nanmean(np.array(sh[g]) / np.array(pn[g]) * 10) for g in sh}
        loo = [(est[g] - np.mean([est[h] - S.true[h] for h in S.NORMAL if h != g]) - S.true[g]) * 1000 for g in S.NORMAL]
        nan = sum(np.isnan(x) for v in sh.values() for x in v)
        Pr(f"{boxname:5s} {vname:12s} px NO4/5/6 {m[4]:.2f}/{m[5]:.2f}/{m[6]:.2f} 기울기 {a*1000:6.2f} | 검증 MAE {mae:4.1f} 정상40장 OK {vok} | NG20장 중 OK {ngok} | 핀고정 LOO {np.mean(np.abs(loo)):5.1f} | 실패 {nan}")
open(os.path.join(S.HERE, 'anchor_result.txt'), 'w').write('\n'.join(out))
