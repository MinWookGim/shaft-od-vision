import pandas as pd, numpy as np
m = pd.read_csv('mid_check.csv'); L = pd.read_csv('light_check.csv', dtype={'time': str})[['file', 'time', 'pin', 'no']]
d = m.merge(L, on='file', how='left')
for k in ['bright', 'vL', 'vR', 'floor']: d[k] = d[[f'{k}_R', f'{k}_G', f'{k}_B']].mean(1); d[k + '_RB'] = d[f'{k}_R'] / d[f'{k}_B']
t5 = d[d.set == 'T5']; b = d[d.set == '1']; r = d[d.set == '2']
print('== 감도: 테스트5(노출 1.00) -> 본시험 NO2 앞 5장(노출 0.83, 17% 적음). 그 사이 빛이 그대로라면 선형이면 -17%')
b2 = b[b.no == 'NO02'].head(5)
for k in ['bright', 'vL', 'vR', 'floor']:
    print(f'  {k:6s} 테스트5 {t5[k].mean():7.2f} -> 본시험 {b2[k].mean():7.2f}  변화 {(b2[k].mean()/t5[k].mean()-1)*100:+.1f}%')
b = b.copy(); b['i'] = range(len(b))
print('\n== 본시험 110장(18분): 처음 10장 vs 마지막 10장, 사진 사이 sd, 핀과 상관(전체 / 같은 시편 안 중앙값)')
for k in ['bright', 'vL', 'vR', 'floor', 'vL_RB', 'vR_RB', 'floor_RB']:
    ch = (b[k].iloc[-10:].mean() / b[k].iloc[:10].mean() - 1) * 100
    within = np.nanmedian([np.corrcoef(g[k], g['pin'])[0, 1] for _, g in b.groupby('no')])
    print(f'  {k:9s} 평균 {b[k].mean():8.3f}  변화 {ch:+.3f}%  sd {b[k].std()/b[k].mean()*100:.3f}%  상관 {np.corrcoef(b[k], b["pin"])[0,1]:+.2f} / {within:+.2f}')
print('  핀 변화', round((b.pin.iloc[-10:].mean() / b.pin.iloc[:10].mean() - 1) * 100, 3), '%')
print('\n== 재장착 30장: 평균과 sd(%)')
for k in ['bright', 'vL', 'vR', 'floor']: print(f'  {k:6s} {r[k].mean():7.2f}  sd {r[k].std()/r[k].mean()*100:.2f}%  핀과 상관 {np.corrcoef(r[k], r["pin"])[0,1]:+.2f}')
# 시간 순서 그림
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'); plt.rcParams['font.family'] = ['NanumGothic']; plt.rcParams['axes.unicode_minus'] = False
s = d[d.set.isin(['1', '2'])].copy(); s['t'] = pd.to_datetime(s['time'], format='%H%M%S')
fig, ax = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
for a, k, lab in zip(ax, ['pin', 'bright', 'vL', 'vR', 'floor'], ['고정 핀 (px)', '밝은 배경', 'V블록 왼쪽 면', 'V블록 오른쪽 면', '바닥 켄트지']):
    v = s[k] if k == 'pin' else s[k] / s[k].iloc[:10].mean() * 100 - 100
    a.plot(s['t'], v, '.', ms=4, color='#d9463e' if k == 'pin' else '#1f5fa8'); a.set_ylabel(lab + ('' if k == 'pin' else '\n(처음 대비 %)'), fontsize=9); a.grid(alpha=.3)
ax[0].set_title('본시험(~15:55)과 재장착(15:57~) 동안 핀 크기와 영역 밝기', fontsize=11)
fig.tight_layout(); fig.savefig('fig_mid_light.png', dpi=110)
