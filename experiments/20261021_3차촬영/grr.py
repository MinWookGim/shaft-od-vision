# MSA 정리: 3차 촬영 데이터로 Gage R&R 과 MSA 다섯 항목(반복성·재현성·치우침·직선성·안정성)을 숫자로 놓는다
# 데이터 구조 (사용 190장 중 암막 있는 150장)
#   본시험: 12시편 x 1번 놓임 x 사진 10장  -> 사진 사이 흔들림(놓임 그대로 셔터만 다시)
#   재장착: NO7~12 x 5번 놓임 x 사진 1장   -> 다시 놓을 때 흔들림 + 사진 흔들림
# 측정자 1명이라 재현성(사람 사이)은 못 구한다 -> GRR 은 반복성만으로 계산하고 '낮게 잡힌 값'임을 표시
# 사용법: python grr.py > grr.txt
import numpy as np, pandas as pd
exec(open('analyze.py').read().split("m['x'] = m[COL]")[0].replace("COL = sys.argv[1] if len(sys.argv) > 1 else 'v3_px'", "COL = 'p50_px'"))
m = m[m.stage != 'noblack'].sort_values('file').copy()
UM = 20.02 / 750 * 1000                 # 물리 배율 약 26.7 µm/px
TOL = 20.0                              # 공차 폭 µm (20.010 ~ 20.030)
PIN_REF = m.pin_px.median()
t = pd.Series(TRUE)
# 시간 보정: 사진마다 핀 값을 그대로 빼면 핀 측정 잡음까지 더해진다 -> 시간 순서로 앞뒤 9장 중앙값(평활)을 써서 흐름만 뺀다
m['pin_s'] = m.pin_px.rolling(9, center=True, min_periods=1).median()
m['p50_corr'] = m.p50_px - (m.pin_s - PIN_REF)
# 본시험 놓임도 '사진 1장짜리 놓임'으로 넣은 판: 시편마다 본시험 첫 장 + 재장착 5장 = 6번 놓임 (본시험-재장착 약 20분 시간차 포함)
m['first'] = m.groupby(['stage', 'no']).cumcount() == 0

def components(col):
    # 사진 흔들림 sd_e: 본시험 12시편 10장씩, 시편 안에서 모은(pooled) sd
    main = m[m.stage == 'main']
    var_e = main.groupby('no')[col].var(ddof=1).mean()
    # 재장착 한 장짜리 값의 시편 안 분산 = 놓임 분산 + 사진 분산
    rem = m[m.stage == 'remount']
    var_r = rem.groupby('no')[col].var(ddof=1).mean()
    var_mount = max(var_r - var_e, 0.0)
    six = m[(m.no >= 7) & ((m.stage == 'remount') | m['first'])]
    var_6 = six.groupby('no')[col].var(ddof=1).mean()
    return np.sqrt(var_e) * UM, np.sqrt(var_mount) * UM, np.sqrt(var_r) * UM, np.sqrt(var_6) * UM

pv = t[[7, 8, 9, 10, 11, 12]].std(ddof=1) * 1000      # 재장착 6시편의 실제 지름 sd (µm)
pv4 = t[[7, 8, 9, 10]].std(ddof=1) * 1000
print(f'물리 배율 {UM:.2f} µm/px, 공차 폭 {TOL:.0f} µm, 재장착 6시편 실측 sd(부품 변동 PV) {pv:.1f} µm\n')

for name, col in [('보정 없음 (50% 방식)', 'p50_px'), ('시간 보정 (핀이 잃은 px 더함, 사후)', 'p50_corr'), ('V3 엔진', 'v3_px')]:
    se, sm, sr, s6 = components(col)
    grr = sr                                          # 한 번 놓고 한 장 찍는 것 = 측정 1회
    print(f'[{name}]')
    print(f'  사진 사이 sd {se:5.1f} µm | 놓임 사이 sd(사진 몫 뺀 것) {sm:5.1f} µm | 측정 1회 sd(=GRR, 재현성 0 가정) {grr:5.1f} µm')
    print(f'  %GRR(공차 대비) = 6 x {grr:.1f} / {TOL:.0f} = {6*grr/TOL*100:4.0f}%   (10% 이하 합격, 30% 넘으면 불합격)')
    print(f'  본시험 첫 장까지 넣은 6번 놓임 sd {s6:5.1f} µm -> %GRR {6*s6/TOL*100:4.0f}%  (본시험과 재장착 사이 약 20분 시간차 포함)')
    print(f'  ndc = 1.41 x PV/GRR = 1.41 x {pv:.1f}/{grr:.1f} = {1.41*pv/grr:.1f}  | 정상 4개(NO7~10)만 PV {pv4:.1f} -> ndc {1.41*pv4/grr:.1f}   (5 이상 필요)')
    print(f'  Type-1 Cg = 0.2 x 공차 / (6 x 사진 sd) = {0.2*TOL/(6*se):.2f}   (1.33 이상 필요, 한 시편 계속 찍기 기준)')
    for k, n in [(1, 1), (10, 1), (10, 6), (1000, 1000)]:
        s = np.sqrt(sm**2 / n + se**2 / (k * n))
        print(f'    놓임 {n:>4}번 x 사진 {k:>4}장 평균하면 sd {s:4.1f} µm -> %GRR {6*s/TOL*100:4.0f}%')
    print()

print(f'30% 안에 들려면 측정 1회 sd <= {0.3*TOL/6:.1f} µm, 10% 면 <= {0.1*TOL/6:.2f} µm')
# V3 는 NO7 재장착 한 장(약 759px)이 튐 -> 그 한 장 빼고 다시
v = m[(m.stage == 'remount')].copy(); v = v[v.v3_px < 755]
print(f'참고: V3 에서 튄 한 장 빼면 재장착 한 장 sd {v.groupby("no").v3_px.var(ddof=1).mean()**0.5*UM:.1f} µm')

# 치우침·직선성: 시간 보정 + 물리 배율 고정 + CAL 3 평균으로 치우침만 맞춘 예측 (drift_removed.py 와 같은 방식)
m['mount'] = np.where(m.stage == 'remount', m.groupby(['stage', 'no']).cumcount() + 1, 0)
mt = m.groupby(['no', 'stage', 'mount']).p50_corr.mean().reset_index()
s = mt.groupby('no').p50_corr.mean()
off = (t[[1, 2, 3]] - s[[1, 2, 3]] * UM / 1000).mean()
e = (off + s * UM / 1000 - t) * 1000
ev = e[[4, 5, 6, 7, 8, 9, 10, 11, 12]]                # CAL 뺀 9개
b, a = np.polyfit(t[ev.index] * 1000 - 20000, ev, 1)
r = np.corrcoef(t[ev.index], ev)[0, 1]
print('\n[치우침·직선성] 시간 보정 후, CAL 3 으로 맞춘 예측의 시편별 오차 (µm)')
print('  ' + ' '.join(f'NO{i}:{e[i]:+.0f}' for i in ev.index))
print(f'  평균 치우침 {ev.mean():+.1f} µm, 시편별 치우침 sd {ev.std(ddof=1):.1f} µm')
print(f'  직선성: 오차 = {a:+.1f} + {b:+.2f} x (실측-20.000mm, µm)   상관 {r:+.2f}  (9개, |r|>0.67 이어야 우연 아님)')

# 안정성: 움직이지 않은 핀의 시간 흐름
main = m[m.stage == 'main']
p0, p1 = main[main.no == 2].pin_px.mean(), main[main.no == 12].pin_px.mean()
print(f'\n[안정성] 고정 핀 NO2 구간 {p0:.2f}px -> NO12 구간 {p1:.2f}px ({(p1/p0-1)*100:+.2f}%, 약 18분), '
      f'Ø20 환산 {(p1/p0-1)*20020:+.0f} µm (공차 폭 {TOL:.0f} µm)')
