# 10-23 멘토링용 그림 2장: (1) 고정 핀 크기의 시간 흐름 (2) 시간 변화 보정 전후 예측 vs 실측
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
OUT = '../../ETC/261023/'
d = pd.read_csv('pin_motion.csv').sort_values('file')
d['t'] = pd.to_datetime(d.file.str[:15], format='%Y%m%d_%H%M%S')
fig, ax = plt.subplots(figsize=(9, 4.2))
for st, c, lb in [(1, '#1f5fa8', '본시험 (시편당 10장 연속, 약 2초 간격)'), (2, '#d9822b', '재장착 (1장씩, 약 2분 간격)')]:
    s = d[d.stage == st]; ax.plot(s.t, s.H, 'o', ms=3.5, color=c, label=lb)
for no, s in d[d.stage == 1].groupby('no'):
    ax.text(s.t.iloc[0], s.H.max() + 0.35, f'NO{no}', fontsize=8, color='#555')
ax.set_ylabel('고정 핀 지름 (px)'); ax.set_xlabel('촬영 시각 (10-21)')
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax.set_title('(NO1은 줌·셔터 설정이 달라 제외) 움직이지 않는 핀이 연속 촬영 중 작아지고, 쉬면 다시 커진다\n본시험 20분 동안 -3.0px (-0.5%, 샤프트로 환산 약 93µm) / 공차 전폭 20µm = 0.75px', fontsize=10.5)
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + '그림1_핀크기_시간흐름.png', dpi=150)

# (2) 보정 전후
exec(open('analyze.py').read().split("m['x'] = m[COL]")[0].replace("COL = sys.argv[1] if len(sys.argv) > 1 else 'v3_px'", "COL = 'p50_px'"))
m = m[m.stage != 'noblack'].sort_values('file').copy(); REF = m.pin_px.median(); UM = 20.02 / 750
t = pd.Series(TRUE)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (nm, xc) in zip(axs, [('보정 없음', m.p50_px), ('핀이 잃은 px만큼 더함 (사후 진단)', m.p50_px - (m.pin_px - REF))]):
    m['xc'] = xc; m['mount'] = np.where(m.stage == 'remount', m.groupby(['stage', 'no']).cumcount() + 1, 0)
    s = m.groupby(['no', 'stage', 'mount']).xc.mean().groupby('no').mean()
    off = (t[[1, 2, 3]] - s[[1, 2, 3]] * UM).mean(); p = off + s * UM
    ax.axhspan(20.010, 20.030, color='#dff0d8'); ax.axvspan(20.010, 20.030, color='#dff0d8', alpha=.5)
    ax.plot([19.95, 20.06], [19.95, 20.06], '--', color='#888', lw=1)
    for i in t.index:
        k = KIND[i]; c = {'CAL': '#1f5fa8', 'VAL': '#2a9d8f', 'RAN': '#2a9d8f', 'NG1': '#c0392b', 'NG2': '#c0392b'}[k]
        ax.plot(t[i], p[i], 'o' if k not in ('NG1', 'NG2') else 'X', color=c, ms=8); ax.text(t[i] + .001, p[i] + .002, f'NO{i}', fontsize=7.5)
    e = (p - t)[4:11] * 1000; r = np.corrcoef(s, t)[0, 1]
    ax.set_title(f'{nm}\n상관 {r:.2f} / 평가 7개 MAE {e.abs().mean():.1f}µm', fontsize=10)
    ax.set_xlabel('업체 실측 (mm)'); ax.set_xlim(19.955, 20.06); ax.set_ylim(*( (19.86, 20.10) if ax is axs[0] else (19.95, 20.07) )); ax.grid(alpha=.3)
axs[0].set_ylabel('사진 예측 (mm), 물리 배율 고정 + CAL 3개로 치우침만 맞춤')
fig.suptitle('초록 띠 = 공차 20.010~20.030. 파랑=CAL, 청록=평가, 빨강 X=NG. 기준선(CAL 평균 부르기) MAE 6.4µm', fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT + '그림2_보정전후_예측vs실측.png', dpi=150)
print('ok')
