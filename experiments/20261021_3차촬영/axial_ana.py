# 3번 분석: Ø20 단 안에서 측정 위치를 바꾸면 지름이 얼마나 달라지나 (국소 50% 방식, axial_profile_local.npz)
# 시편 공통 모양(렌즈·조명 탓일 수 있음)은 빼고, 시편마다 다른 모양(진짜 테이퍼 후보)만 본다
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumGothic' in f: fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
TRUE = {1: 20.011, 2: 20.020, 3: 20.028, 4: 20.019, 5: 20.030, 6: 20.011, 7: 20.014, 8: 20.026, 9: 20.009, 10: 20.022, 11: 19.968, 12: 20.044}
UM = 26.6
z = np.load('axial_profile_local.npz'); xs = z['xs']; D = z['d']; files = z['files']
loc = pd.read_csv('local_edges.csv').set_index('file')
no = loc.loc[files, 'no'].str[2:].astype(int).values; st = loc.loc[files, 'stage'].values
pin = loc.loc[files, 'pin'].values
in_step = (xs >= 8015) & (xs <= 8335)
def nfit(x, y):
    # NaN 빼고 직선 기울기 (몇몇 열에서 경계를 못 잡아 NaN)
    k = np.isfinite(y); return np.polyfit(x[k], y[k], 1)[0]

# 시편별 본시험 10장 평균 지름 분포
prof = {i: np.nanmean(D[(no == i) & (st == 1)], axis=0) for i in range(1, 13)}
P = np.array([prof[i] for i in range(1, 13)])
common = np.nanmean(P, axis=0)
print('[1] 시편 공통 모양 (12시편 평균, Ø20 단 안): 홈 쪽 끝 -> 칼라 쪽 끝 기울기 %.2f px (%.1f µm), 열마다 흔들림 범위 %.2f px'
      % (nfit(xs[in_step], common[in_step]) * 320, nfit(xs[in_step], common[in_step]) * 320 * UM,
         np.nanmax(common[in_step]) - np.nanmin(common[in_step])))

# 시편마다 다른 모양: 공통 모양 뺀 뒤 단 안에서 직선 기울기(테이퍼) = 320px(약 8.5mm) 동안 지름 변화
print('\n[2] 시편마다 다른 테이퍼 (공통 모양 뺀 것, 단 길이 약 8.5mm 동안 지름 변화, µm)')
tap = {}
for i in range(1, 13):
    r = P[i - 1] - common
    k = nfit(xs[in_step], r[in_step])
    tap[i] = k * 320 * UM
# 재장착 사진으로 같은 시편 테이퍼가 다시 나오나 (진짜 모양이면 반복돼야 함)
rep = {}
for i in range(7, 13):
    rr = np.nanmean(D[(no == i) & (st == 2)], axis=0) - common
    rep[i] = nfit(xs[in_step], rr[in_step]) * 320 * UM
print('  본시험: ' + ' '.join(f'NO{i}:{tap[i]:+.1f}' for i in range(1, 13)))
print('  재장착: ' + ' '.join(f'NO{i}:{rep[i]:+.1f}' for i in range(7, 13)))
a = np.array([tap[i] for i in range(7, 13)]); b = np.array([rep[i] for i in range(7, 13)])
print(f'  시편 사이 테이퍼 sd {np.std(list(tap.values()), ddof=1):.1f} µm | 본시험 vs 재장착 상관 {np.corrcoef(a, b)[0,1]:+.2f} (6개, |r|>0.81 이어야 우연 아님)')

# 사진 한 장 안 테이퍼 잡음 크기: 본시험 같은 시편 10장 사이 테이퍼 sd
ts = []
for i in range(1, 13):
    for d in D[(no == i) & (st == 1)]:
        ts.append((i, nfit(xs[in_step], (d - common)[in_step]) * 320 * UM))
ts = pd.DataFrame(ts, columns=['no', 'tap'])
print(f'  같은 놓임 사진 사이 테이퍼 sd {ts.groupby("no").tap.std().mean():.1f} µm -> 10장 평균의 잡음 약 {ts.groupby("no").tap.std().mean()/np.sqrt(10):.1f} µm')

# 측정 위치별 시편 오차: 단을 세 칸(홈 쪽/가운데/칼라 쪽)으로 나눠 각각 같은 방식(같은 px 시간 보정, 물리 배율, CAL3 치우침)으로
print('\n[3] 측정 위치를 바꾸면 시편별 오차가 달라지나 (본시험 10장, 같은 px 시간 보정 + 물리 배율 + CAL3 치우침)')
PIN_REF = np.median(pin)
cuts = {'홈 쪽 1/3': (8015, 8122), '가운데 1/3': (8122, 8228), '칼라 쪽 1/3': (8228, 8335), '전체': (8015, 8335)}
t = pd.Series(TRUE); out = {}
for nm, (x0, x1) in cuts.items():
    w = (xs >= x0) & (xs < x1)
    v = np.nanmedian(D[:, w], axis=1) - (pin - PIN_REF)
    s = pd.Series(v[st == 1]).groupby(no[st == 1]).mean()
    off = (t[[1, 2, 3]] - s[[1, 2, 3]] * UM / 1000).mean()
    e = (off + s * UM / 1000 - t) * 1000; out[nm] = e
    ev = e[4:11]; nor = e[e.index <= 10]
    print(f'  {nm:10s} 평가7 MAE {ev.abs().mean():5.1f} | 정상10 치우침 sd {nor.std(ddof=1):5.1f} | NG1 {e[11]:+6.1f} NG2 {e[12]:+6.1f} µm')
o = pd.DataFrame(out).round(1)
print(o.T.to_string())
# 위치 차(칼라 쪽 - 홈 쪽)가 진짜 모양이면 재장착에서도 반복돼야 한다 + 사진 한 장 잡음 크기
def third_diff(d):
    a = np.nanmedian(d[(xs >= 8228) & (xs < 8335)]); b = np.nanmedian(d[(xs >= 8015) & (xs < 8122)]); return (a - b) * UM
td = pd.DataFrame({'no': no, 'st': st, 'td': [third_diff(d) for d in D]})
g = td.groupby(['no', 'st']).td.agg(['mean', 'std', 'size']).unstack()
print('  위치 차(칼라 쪽 - 홈 쪽, µm) 본시험 10장 평균 vs 재장착 5장 평균:')
print('   ' + ' '.join(f'NO{i}:{g.loc[i, ("mean", 1)]:+.0f}/{g.loc[i, ("mean", 2)]:+.0f}' for i in range(7, 13)))
mm, rr2 = g.loc[7:12, ('mean', 1)].values, g.loc[7:12, ('mean', 2)].values
print(f'   본시험 vs 재장착 상관 {np.corrcoef(mm, rr2)[0,1]:+.2f} (6개) | 사진 한 장 위치차 sd {g[("std", 1)].mean():.1f} µm -> 10장 평균 잡음 약 {g[("std", 1)].mean()/np.sqrt(10):.1f} µm')
print(f'   12시편 위치차(본시험) 평균 {g[("mean", 1)].mean():+.1f} sd {g[("mean", 1)].std():.1f} µm  <- 공통 부분(렌즈·조명 몫 포함)을 뺀 시편 사이 퍼짐')
print('  위치 사이 차(칼라 쪽 - 홈 쪽) 시편별: ' + ' '.join(f'NO{i}:{o.loc[i,"칼라 쪽 1/3"]-o.loc[i,"홈 쪽 1/3"]:+.0f}' for i in range(1, 13)))

# 그림
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for i in range(1, 13):
    ax[0].plot(xs, (P[i - 1]) * UM / 1000, lw=0.8, label=f'NO{i}')
ax[0].axvspan(8015, 8335, color='0.9', zorder=0); ax[0].set_ylim(19.9, 20.25)
ax[0].set_title('시편별 지름 분포 (본시험 10장 평균, px x 26.6µm, 절대값은 의미 없음)'); ax[0].set_xlabel('사진 가로 위치 (px, 왼쪽=홈 쪽)'); ax[0].set_ylabel('mm 환산')
for i in range(1, 13):
    ax[1].plot(xs[in_step], (P[i - 1] - common)[in_step] * UM, lw=0.8)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_title('공통 모양 뺀 시편별 차이 (µm)'); ax[1].set_xlabel('사진 가로 위치 (px)'); ax[1].set_ylabel('µm')
ax[0].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig('axial_profile.png', dpi=110)
