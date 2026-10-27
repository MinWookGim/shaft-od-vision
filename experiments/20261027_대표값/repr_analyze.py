# 2단계: 대표값 후보별로 시편 지름(px) -> mm, 실측과 비교
# 후보는 3회차 멘토링 PDF에 선행 개발자가 미리 적어 둔 것 전부 + 멘토가 말한 경계 envelope. 결과를 보고 고르지 않도록 전부 보고한다
# 교정: 물리 배율 고정(핀 Ø16 기준) + 치우침만 맞춤. (A) CAL(NO1~3)로 맞추고 평가 7개(NO4~10), (B) 정상 10개 하나씩 빼고 맞히기
# 시간 보정: 핀이 잃은 px만큼 샤프트에 더함(3차 분석정리 5절과 같은 방식). 보정 전·후 둘 다
import numpy as np, pandas as pd
d = pd.read_csv('repr_cols.csv'); pin = pd.read_csv('../1021_main/local_edges.csv')[['file', 'pin']]
d = d.merge(pin, on='file')
T = d[[f't{i}' for i in range(31)]].to_numpy(); B = d[[f'b{i}' for i in range(31)]].to_numpy(); D = B - T
def topk(a, k): return np.sort(a)[-k:].mean()
C = {
    'Median': lambda D, T, B: np.nanmedian(D),
    'Mean': lambda D, T, B: np.nanmean(D),
    'P90': lambda D, T, B: np.nanpercentile(D, 90),
    'P95': lambda D, T, B: np.nanpercentile(D, 95),
    '상위3 평균': lambda D, T, B: topk(D[~np.isnan(D)], 3),
    '상위5 평균': lambda D, T, B: topk(D[~np.isnan(D)], 5),
    'Max': lambda D, T, B: np.nanmax(D),
    'envelope(위·아래 끝점)': lambda D, T, B: np.nanmax(B) - np.nanmin(T),
}
for k, f in C.items(): d[k] = [f(D[i], T[i], B[i]) for i in range(len(d))]
mad = np.nanmedian(np.abs(D - np.nanmedian(D, 1)[:, None]), 1) * 1.4826
d['튀는 열 수'] = (D > np.nanmedian(D, 1)[:, None] + 3 * mad[:, None]).sum(1)
d['열 sd(µm)'] = np.nanstd(D, 1)
true = {'NO01': 20.011, 'NO02': 20.020, 'NO03': 20.028, 'NO04': 20.019, 'NO05': 20.030, 'NO06': 20.011, 'NO07': 20.014,
        'NO08': 20.026, 'NO09': 20.009, 'NO10': 20.022, 'NO11': 19.968, 'NO12': 20.044}
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
if MODE == 'main':      # 본시험 10장만, 핀 기준 배율
    m = d[d.stage == 1].copy()
else:                   # 3차 분석정리 5절과 같은 설정: 본시험+재장착, 놓임별 평균 후 시편 평균, 배율 20.02/750mm
    m = d.copy()
pin_ref = m[m.no == 'NO02'].pin.mean(); s = 16 / m.pin.mean() if MODE == 'main' else 20.02 / 750
um = s * 1000
print(f'배율 {um:.2f}µm/px, 핀 기준 NO2 {pin_ref:.2f}px')
print(f"한 장 안 31개 열: 열 sd 중앙값 {np.nanmedian(m['열 sd(µm)'])*um:.1f}µm, 튀는 열(중앙값+3MAD 초과) 장당 평균 {m['튀는 열 수'].mean():.2f}개, 최대 {m['튀는 열 수'].max()}개")
normal = [f'NO{i:02d}' for i in range(1, 11)]; cal = ['NO01', 'NO02', 'NO03']; ev = [f'NO{i:02d}' for i in range(4, 11)]
rows = []
for corr in [False, True]:
    for k in C:
        px = m[k] + ((pin_ref - m.pin) if corr else 0)
        if MODE == 'main':
            spx = px.groupby(m.no).median()
        else:
            mount = np.where(m.stage == 2, m.groupby(['stage', 'no']).cumcount() + 1, 0)
            spx = px.groupby([m.no, mount]).mean().groupby(level=0).mean()
        rep = (px[m.stage == 1] * um).groupby(m.no[m.stage == 1]).std().median()
        y = pd.Series(true); mm = s * spx
        off = (y[cal] - mm[cal]).mean(); pred = mm + off
        eA = (pred[ev] - y[ev]) * 1000
        eB = pd.Series({n: ((mm[n] + (y[[o for o in normal if o != n]] - mm[[o for o in normal if o != n]]).mean()) - y[n]) * 1000 for n in normal})
        ng = ' / '.join(f'{pred[n]:.4f}' for n in ['NO11', 'NO12'])
        rows.append(dict(시간보정='후' if corr else '전', 대표값=k, 사진사이sd=round(rep, 1), A_MAE=round(eA.abs().mean(), 1), A_최대=round(eA.abs().max(), 1),
                         B_MAE=round(eB.abs().mean(), 1), B_RMS=round(np.sqrt((eB ** 2).mean()), 1), NG예측=ng, _eB=eB))
y = pd.Series(true)
baseA = (y[cal].mean() - y[ev]).abs().mean() * 1000
baseB = pd.Series({n: (y[[o for o in normal if o != n]].mean() - y[n]) for n in normal}).abs().mean() * 1000
r = pd.DataFrame(rows)
print(f'\n기준선(측정 없이 평균 부르기): A MAE {baseA:.1f}µm, B MAE {baseB:.1f}µm')
print(r.drop(columns='_eB').to_string(index=False))
# Median 대비 각 후보의 시편별 절대오차 차이(B, 시간 보정 후): 몇 개 시편에서 나은가
print('\n시간 보정 후, 하나씩 빼고 맞히기(B)에서 Median 대비 (시편 10개 중 더 나은 개수, 평균 절대오차 차이 µm)')
med = r[(r.시간보정 == '후') & (r.대표값 == 'Median')]._eB.iloc[0].abs()
for _, q in r[r.시간보정 == '후'].iterrows():
    e = q._eB.abs(); print(f"  {q.대표값:22s} 나은 시편 {(e < med).sum():2d}/10, 차이 {(e - med).mean():+.1f}µm")
r.drop(columns='_eB').to_csv(f'repr_summary_{MODE}.csv', index=False)
