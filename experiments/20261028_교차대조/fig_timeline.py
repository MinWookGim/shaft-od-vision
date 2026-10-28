# 그림: 3차 본시험·재장착 시간축에 고정 핀 지름, 핀 경계 흐림 폭, 역광판 질감 상관을 세 칸으로 나란히
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
w = pd.read_csv('edge_width.csv'); t = pd.read_csv('timeline.csv')[['file', 'pin', 'min']]
a = pd.read_csv('texture_aligned.csv')[['file', 'c_ref']]
d = w.merge(t, on='file').merge(a, on='file')
g = d.groupby(['stage', 'no'], sort=False).agg(min=('min', 'mean'), pin=('pin', 'mean'), wt=('pin_wt', 'mean'), wb=('pin_wb', 'mean'), c=('c_ref', 'mean')).reset_index()
B, O, INK, MUTED = '#2a78d6', '#eb6834', '#2b2b2a', '#8a8a85'
fig, ax = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw=dict(hspace=0.18))
r0 = d[d.stage == 2]['min'].min() - 1
for x in ax:
    x.axvspan(r0, d['min'].max() + 1, color='#f0efec', zorder=0)
    x.axvline(28.3, color=MUTED, lw=1, ls=':', zorder=1)
    for s in ['top', 'right']: x.spines[s].set_visible(False)
    x.grid(axis='y', color='#e6e5e0', lw=0.8); x.tick_params(colors=INK, labelsize=9)
ax[0].scatter(d['min'], d.pin, s=10, color=B, alpha=0.35, lw=0)
ax[0].plot(g['min'], g.pin, color=B, lw=2, marker='o', ms=4)
ax[0].set_ylabel('고정 핀 지름 (px)', color=INK, fontsize=10)
ax[0].text(0.5, 614.75, '본시험 (NO2 -> NO12, 10장씩)', fontsize=9, color=MUTED)
ax[0].text(r0 + 0.8, 614.75, '재장착 (NO7 -> NO12)', fontsize=9, color=MUTED)
ax[0].set_ylim(612.6, 614.95)
ax[1].plot(g['min'], g.wb, color=B, lw=2, marker='o', ms=4); ax[1].plot(g['min'], g.wt, color=O, lw=2, marker='o', ms=4)
ax[1].text(g['min'].iloc[-1] + 0.6, g.wb.iloc[-1], '아래 경계', color=INK, fontsize=9, va='center')
ax[1].text(g['min'].iloc[-1] + 0.6, g.wt.iloc[-1], '위 경계', color=INK, fontsize=9, va='center')
ax[1].set_ylabel('핀 경계 흐림 폭\n10~90% (px)', color=INK, fontsize=10)
ax[2].plot(g['min'], g.c, color=B, lw=2, marker='o', ms=4)
ax[2].axhline(0.80, color=MUTED, lw=1, ls='--'); ax[2].text(0.3, 0.815, '바로 앞 사진과의 상관(사진 잡음 상한) 0.80', fontsize=8.5, color=MUTED)
ax[2].set_ylabel('역광판 질감 상관\n(이동 맞춘 뒤)', color=INK, fontsize=10); ax[2].set_ylim(0.3, 0.9)
ax[2].text(33.5, 0.60, 'NO8 -> NO9: 질감만 약 232px 이동\n(고정 물체는 3px 안)', fontsize=8.5, color=INK)
ax[2].set_xlabel('첫 사진부터 지난 시간 (분)', color=INK, fontsize=10); ax[2].set_xlim(-1, 47.5)
for x in ax: x.yaxis.set_label_coords(-0.075, 0.5)
fig.suptitle('3차 촬영 중 고정 핀이 작아질 때 경계가 흐려졌고, 역광판이 움직인 뒤 함께 돌아왔다', fontsize=12, color=INK, x=0.07, ha='left', y=0.95)
fig.savefig('fig_timeline.png', dpi=140, bbox_inches='tight'); print('saved')
