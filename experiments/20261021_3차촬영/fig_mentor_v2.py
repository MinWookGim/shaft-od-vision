# 10-23 새벽: 경계 정의 결함 발견 후 그림 v2 (원본 그림1·2 는 그대로 둠)
# 그림1 v2: 고정 핀 크기 변화를 두 경계 정의로 같이 (NO2 평균 대비 %)
# 그림2 v2: 국소 50% 방식 값으로 보정 전후 예측 vs 실측
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
OUT = '../../ETC/261023/'
l = pd.read_csv('local_edges.csv').merge(pd.read_csv('v3_d20.csv')[['file', 'pin_px']], on='file')
l['no'] = l['no'].str[2:].astype(int); l['t'] = pd.to_datetime(l.file.str[:15], format='%Y%m%d_%H%M%S')
l = l[(l.stage < 3) & ~((l.stage == 1) & (l.no == 1))].sort_values('t')
fig, ax = plt.subplots(figsize=(9, 4.4))
for col, c, lb in [('pin_px', '#999999', '기존 방식 (위 경계를 어두운 바닥 쪽에서 잡음, 결함)'), ('pin', '#1f5fa8', '국소 50% 방식 (경계 바로 안팎 밝기의 중간)')]:
    ref = l[(l.stage == 1) & (l.no == 2)][col].mean()
    for st, mk in [(1, 'o'), (2, 's')]:
        s = l[l.stage == st]
        ax.plot(s.t, (s[col] / ref - 1) * 100, mk, ms=3.5 if st == 1 else 5, color=c, label=lb if st == 1 else None)
ax.axhline(0, color='k', lw=.5)
ax.axhspan(-0.1, 0, color='#dff0d8', alpha=.6)
ax.text(l.t.iloc[0], -0.095, '공차 전폭 20µm 에 해당하는 크기 (0.1%)', fontsize=8, color='#2e7d32')
ax.set_ylabel('고정 핀 크기 변화 (NO2 대비 %)'); ax.set_xlabel('촬영 시각 (10-21). 동그라미=본시험 연속 10장, 네모=재장착 1장씩')
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax.set_title('움직이지 않는 핀이 본시험 동안 작아지고 재장착 때 회복. 크기는 경계 정의에 따라 -0.46% / -0.18%\n'
             '(어느 쪽이든 공차보다 큼. 정의에 따라 크기가 달라진다 = 경계 밝기 모양 자체가 변했다)', fontsize=10)
ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + '그림1_v2_핀크기_두경계정의.png', dpi=150)

os.environ['SRC'] = 'v3_d20_local.csv'
src = open('fig_mentor.py').read().split('# (2) 보정 전후')[1]
src = src.replace("'그림2_보정전후_예측vs실측.png'", "'그림2_v2_보정전후_국소50.png'").replace("기준선(CAL 평균 부르기) MAE 6.4µm'", "기준선(CAL 평균 부르기) MAE 6.4µm. 국소 50% 경계'")
src = src.replace("(19.86, 20.10) if ax is axs[0] else (19.95, 20.07)", "(19.90, 20.08) if ax is axs[0] else (19.93, 20.07)")
exec(src)
