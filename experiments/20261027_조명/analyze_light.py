# light_check.csv 분석: 배경 밝기·색 비율·줄무늬가 시간에 따라 변했나, 핀 px 변화와 같이 움직이나
import pandas as pd, numpy as np
d = pd.read_csv('light_check.csv', dtype={'time': str})
d['t'] = pd.to_datetime('2026-10-21 ' + d['time'], format='%Y-%m-%d %H%M%S')
d['min'] = (d['t'] - d['t'].min()).dt.total_seconds() / 60
for st, nm in [(1, '본시험(연속 10장씩)'), (2, '재장착(약 2분 간격)')]:
    x = d[d.stage == st]
    print(f'== {nm}, {len(x)}장, {x["min"].min():.1f}~{x["min"].max():.1f}분')
    for c in ['bgR', 'bgG', 'bgB', 'gapG', 'R_B', 'band_sd', 'pin']:
        sl = np.polyfit(x['min'], x[c], 1)[0]
        print(f'  {c:8s} 평균 {x[c].mean():8.3f}  범위 {x[c].min():8.3f}~{x[c].max():8.3f}  sd {x[c].std():.3f}  분당 기울기 {sl:+.4f}  핀과 상관 {np.corrcoef(x[c], x["pin"])[0,1]:+.2f}')
b = d[d.stage == 1]
print('본시험 핀 변화 %:', round((b.pin.iloc[-10:].mean() / b.pin.iloc[:10].mean() - 1) * 100, 3))
print('본시험 R/B 변화 %:', round((b.R_B.iloc[-10:].mean() / b.R_B.iloc[:10].mean() - 1) * 100, 3))
print('본시험 배경G 변화 %:', round((b.bgG.iloc[-10:].mean() / b.bgG.iloc[:10].mean() - 1) * 100, 3))
print('줄무늬 주기(px) 분포:', d.band_peak_period.round(0).value_counts().head(5).to_dict())
# 시편 안(같은 시편 연속 10장) 상관: 시편 차이를 빼고 사진 사이 변화끼리
w = b.groupby('no').transform(lambda s: s - s.mean(numeric_only=True)) if False else None
rs = []
for no, g in b.groupby('no'):
    if len(g) > 3: rs.append([np.corrcoef(g[c], g['pin'])[0, 1] for c in ['bgG', 'R_B', 'band_sd']])
print('시편 안 상관 중앙값 (배경G, R/B, 줄무늬):', np.round(np.nanmedian(np.array(rs), axis=0), 2))
