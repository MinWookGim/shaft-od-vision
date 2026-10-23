# 3차 촬영 분석 (시나리오 문서 부록 B 사전 규칙 기준)
# 입력: v3_d20.csv (Ø20 단, V3 px + 내 50% px + 핀 px), 분류 폴더(사용 190장)
# 사용법: python analyze.py [v3_px|p50_px]
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

COL = sys.argv[1] if len(sys.argv) > 1 else 'v3_px'
SHARE = Path('../../field/photos/20261021_원전기계/3차_261021_분류')
TRUE = {1: 20.011, 2: 20.020, 3: 20.028, 4: 20.019, 5: 20.030, 6: 20.011, 7: 20.014, 8: 20.026,
        9: 20.009, 10: 20.022, 11: 19.968, 12: 20.044}
KIND = {1: 'CAL', 2: 'CAL', 3: 'CAL', 4: 'VAL', 5: 'VAL', 6: 'VAL', 7: 'RAN', 8: 'RAN', 9: 'RAN', 10: 'RAN',
        11: 'NG1', 12: 'NG2'}
LO, HI = 20.010, 20.030
PIN_MM = 16.0   # 정식 핀게이지 아님(공칭 Ø16). 절대값은 CAL 로 맞추므로 비율만 쓴다

# 사용 사진 = 분류 폴더에 들어간 190장. 단계·시편은 폴더 이름에서
rows = []
for p in SHARE.glob('*/NO*/*.jpg'):
    stage = {'1': 'main', '2': 'remount', '3': 'noblack'}[p.parts[-3][0]]
    rows.append({'file': p.name, 'stage': stage, 'no': int(p.parts[-2][2:])})
use = pd.DataFrame(rows)
SRC = os.environ.get('SRC', 'v3_d20.csv')   # 국소 경계 재측정판은 SRC=v3_d20_local.csv
m = pd.read_csv(SRC).merge(use, on='file')
assert len(m) == 190, len(m)
m['x'] = m[COL]
m['ratio'] = m['x'] / m['pin_px']
m = m.sort_values('file').reset_index(drop=True)

# 놓임(mount) 번호: 본시험 10장 = 1번 놓임, 재장착 5장 = 각자 한 번씩 놓임, 암막없음 10장 = 1번 놓임
m['mount'] = np.where(m.stage == 'remount', m.groupby(['stage', 'no']).cumcount() + 1, 0)
bk = m[m.stage != 'noblack']
mounts = bk.groupby(['no', 'stage', 'mount']).agg(x=('x', 'mean'), ratio=('ratio', 'mean'), n=('x', 'size'),
                                                  sd=('x', 'std')).reset_index()
spec = mounts.groupby('no').agg(x=('x', 'mean'), ratio=('ratio', 'mean'), n_mount=('x', 'size')).reset_index()
main = mounts[mounts.stage == 'main'].set_index('no')
spec['x_main'] = spec.no.map(main.x)
spec['true'] = spec.no.map(TRUE); spec['kind'] = spec.no.map(KIND)

print(f'=== 측정 열: {COL} ===')
print('\n[1] 시편별 대표값 (px). x=놓임 평균의 평균, x_main=본시험 놓임만, sd_photo=본시험 10장 sd')
spec['sd_photo'] = spec.no.map(main.sd)
print(spec[['no', 'kind', 'true', 'x_main', 'x', 'n_mount', 'sd_photo', 'ratio']].round(4).to_string(index=False))

cal = spec[spec.kind == 'CAL'].copy()
cal_x = main.loc[[1, 2, 3], 'x'].values           # 규칙: CAL 은 본시험 3시편 x 10장
cal_r = main.loc[[1, 2, 3], 'ratio'].values
cal_t = np.array([TRUE[i] for i in (1, 2, 3)])
ev = spec[spec.no.between(4, 10)].copy()          # 평가: VAL 3 + RAN 4
ng = spec[spec.no >= 11].copy()
um_per_px = 20.02 / spec.x.mean() * 1000          # 물리 배율(µm/px) 대략값

def report(name, f):
    e = (np.asarray(f(ev)) - ev.true.values) * 1000
    g = pd.Series(np.asarray(f(ng)))
    calls = ['불합격' if (v < LO or v > HI) else '합격' for v in g]
    print(f'  {name:32s} MAE {np.abs(e).mean():5.1f}  최대 {np.abs(e).max():5.1f}  '
          f'평균치우침 {e.mean():+5.1f} µm | NG1 {g.iloc[0]:.4f}({calls[0]}) NG2 {g.iloc[1]:.4f}({calls[1]})')
    return e

print(f'\n[2] mm 환산 방식별 평가 (VAL 3 + RAN 4 = 7개). 물리 배율 약 {um_per_px:.2f} µm/px')
res = {}
b, a = np.polyfit(cal_x, cal_t, 1)
print(f'  (가) CAL 3점 회귀 기울기 {b*1000:.2f} µm/px (물리 배율 {um_per_px:.2f} 와 비교)')
res['(가) CAL 3점 회귀 [규칙]'] = report('(가) CAL 3점 회귀 [규칙]', lambda d: a + b * d.x)
b2, a2 = np.polyfit(cal_x[1:], cal_t[1:], 1)
print(f'  NO1 뺀 2점 회귀 기울기 {b2*1000:.2f} µm/px')
res['NO1 뺀 2점 회귀'] = report('NO1 뺀 2점 회귀', lambda d: a2 + b2 * d.x)
k = um_per_px / 1000
for nm, idx in [('물리 배율 + CAL 3 평균 치우침', [0, 1, 2]), ('물리 배율 + CAL 2(NO1 뺌) 치우침', [1, 2])]:
    off = np.mean(cal_t[idx] - k * cal_x[idx])
    res[nm] = report(nm, lambda d, off=off: off + k * d.x)
# (나) 핀 비율: d = c * (샤프트px / 핀px), c 는 CAL 평균에 맞춤 (진단용)
c = cal_t.mean() / cal_r.mean()
print(f'  (나) 핀 비율 상수 c = {c:.4f} mm  (핀이 정확히 Ø{PIN_MM} 라면 {PIN_MM:.1f} 이어야 함)')
res['(나) 핀 비율 [진단]'] = report('(나) 핀 비율 [진단]', lambda d: c * d.ratio)
res['기준선: CAL 실측 평균 부르기'] = report('기준선: CAL 실측 평균 부르기', lambda d: np.full(len(d), cal_t.mean()))

print('\n[3] 시편별 오차 (µm)')
tab = pd.DataFrame({k2: np.asarray(v) for k2, v in res.items()}, index=[f'NO{i} {KIND[i]} {TRUE[i]}' for i in ev.no])
print(tab.round(1).to_string())

print('\n[4] Q3 재장착: 사진 사이 sd vs 놓임 사이 sd (px, µm)')
within = main.sd.loc[[7, 8, 9, 10, 11, 12]]
for i in range(7, 13):
    ms = mounts[mounts.no == i].x.values
    print(f'  NO{i}: 본시험 10장 sd {within[i]:.3f}px  | 놓임 6개 평균 {ms.mean():.2f} sd {ms.std(ddof=1):.3f}px '
          f'({ms.std(ddof=1)*um_per_px:.1f} µm) 범위 {ms.min():.2f}~{ms.max():.2f}  본시험-재장착평균 {ms[0]-ms[1:].mean():+.2f}px')
print(f'  평균: 사진 사이 sd {main.sd.mean():.3f}px ({main.sd.mean()*um_per_px:.1f} µm), '
      f'놓임 사이 sd {mounts[mounts.no>=7].groupby("no").x.std().mean():.3f}px '
      f'({mounts[mounts.no>=7].groupby("no").x.std().mean()*um_per_px:.1f} µm)')

print('\n[5] Q4 시간 흐름: 핀 px (움직이지 않은 물체) 시편 순서대로')
pt = bk.groupby(['stage', 'no'], sort=False).agg(t=('file', 'first'), pin=('pin_px', 'mean'), pin_sd=('pin_px', 'std'),
                                                 x=('x', 'mean')).reset_index().sort_values('t')
print(pt.round(3).to_string(index=False))

print('\n[6] 암막 없음 세트 (초점 새로 잡음 -> px 직접 비교 불가, 핀 비율만 비교)')
nb = m[m.stage == 'noblack'].groupby('no').agg(x=('x', 'mean'), sd=('x', 'std'), pin=('pin_px', 'mean'),
                                                 ratio=('ratio', 'mean')).reset_index()
nb['pin_est'] = c * nb.ratio
nb['err_um'] = (nb.pin_est - nb.no.map(TRUE)) * 1000
nb['암막 비율'] = nb.no.map(spec.set_index('no').ratio)
print(nb.round(4).to_string(index=False))
