# 4번 분석: 국소 50% 방식(local_edges.csv, 사용 190장) 기준
# (1) 기존 방식(위 경계를 바닥에서 잡던 v3_d20.csv) vs 국소 방식: 고정 핀이 시간에 따라 얼마나 줄었나
# (2) 암막 없음 40장(앞빛, ISO 12, 초점 새로 잡음): 핀이 줄어드나
# (3) 암막 없음 NO7~10 샤프트 px 와 실측 (시편 4개, 참고만)
import numpy as np, pandas as pd
TRUE = {7: 20.014, 8: 20.026, 9: 20.009, 10: 20.022}
l = pd.read_csv('local_edges.csv'); v = pd.read_csv('v3_d20.csv')
l['no'] = l['no'].str[2:].astype(int)
l['t'] = pd.to_datetime(l.file.str[:15], format='%Y%m%d_%H%M%S')
l = l.merge(v[['file', 'pin_px', 'p50_px']], on='file').sort_values('t')

def slope_min(d, col):
    mins = (d.t - d.t.iloc[0]).dt.total_seconds() / 60
    return np.polyfit(mins, d[col], 1)[0], mins.iloc[-1]

def burst(d, col):
    g = d.groupby('no')[col].agg(['first', 'last']); x = g['last'] - g['first']
    return x.mean(), x.std() / np.sqrt(len(x))

mb = l[(l.stage == 1) & (l.no >= 2)]       # 본시험 NO2~12 (NO1 은 줌)
rm = l[l.stage == 2]
nb = l[l.stage == 3]
print('[1] 고정 핀 시간 흐름: 기존 방식 vs 국소 방식 (px)')
for nm, col in [('기존(위 경계 바닥)', 'pin_px'), ('국소 50%', 'pin')]:
    k, T = slope_min(mb, col); a = mb[mb.no == 2][col].mean(); b = mb[mb.no == 11][col].mean()
    bm, bs = burst(mb, col)
    print(f'  {nm:12s} NO2 {a:.2f} -> NO11 {b:.2f} ({(b/a-1)*100:+.2f}%, Ø20 환산 {(b/a-1)*20020:+.0f}µm) | 기울기 {k:+.3f}px/분 | '
          f'연속 10장 첫->끝 {bm:+.2f}±{bs:.2f}px | 재장착 단계 평균 {rm[col].mean():.2f}')
# 위·아래 경계를 따로: 국소 방식에서 핀 위/아래 경계가 어느 쪽으로 움직였나
a, b = mb[mb.no == 2], mb[mb.no == 11]
print(f'  국소 방식 핀 위 경계 {a.pin_top.mean():.2f} -> {b.pin_top.mean():.2f} ({b.pin_top.mean()-a.pin_top.mean():+.2f}), '
      f'아래 경계 {a.pin_bot.mean():.2f} -> {b.pin_bot.mean():.2f} ({b.pin_bot.mean()-a.pin_bot.mean():+.2f})  (+ = 사진 아래쪽)')

print('\n[2] 암막 없음 40장 (NO7~10)')
print(nb.groupby('no').agg(pin=('pin', 'mean'), pin_sd=('pin', 'std'), shaft=('shaft', 'mean'), shaft_sd=('shaft', 'std')).round(3).to_string())
k, T = slope_min(nb, 'pin'); bm, bs = burst(nb, 'pin')
kb, _ = slope_min(mb, 'pin'); bbm, bbs = burst(mb, 'pin')
print(f'  핀 기울기 {k:+.3f} px/분 ({T:.1f}분), 연속 10장 첫->끝 {bm:+.2f}±{bs:.2f}px')
print(f'  같은 잣대 암막 본시험: 기울기 {kb:+.3f} px/분, 연속 10장 첫->끝 {bbm:+.2f}±{bbs:.2f}px')
print(f'  {T:.1f}분이면 암막 본시험 기울기로 {kb*T:+.2f}px 변했어야 함 (핀 크기 비율로 암막없음 배율 환산 {kb*T*628.8/613.9:+.2f}) vs 실제 {k*T:+.2f}px')
print(f'  사진 사이 sd: 암막 없음 샤프트 {nb.groupby("no").shaft.std().mean():.3f}px vs 암막 본시험 {mb.groupby("no").shaft.std().mean():.3f}px')

print('\n[3] 암막 없음 샤프트 px vs 실측 (시편 4개, 참고)')
t = pd.Series(TRUE)
um_nb = 20.02 / nb.shaft.mean() * 1000; um_b = 20.02 / mb.shaft.mean() * 1000
PIN_REF = l[l.stage != 3].pin.median()
mbc = mb.assign(c=mb.shaft - (mb.pin - PIN_REF))
for nm, s, u in [('암막 없음', nb.groupby('no').shaft.mean(), um_nb),
                 ('암막 본시험 보정 없음', mb.groupby('no').shaft.mean()[[7, 8, 9, 10]], um_b),
                 ('암막 본시험 시간 보정', mbc.groupby('no').c.mean()[[7, 8, 9, 10]], um_b)]:
    rr = s * u / 1000 - t; rr = (rr - rr.mean()) * 1000
    print(f'  {nm:14s} 평균 뺀 시편별 치우침(µm): ' + ' '.join(f'NO{i}:{rr[i]:+5.1f}' for i in t.index) +
          f' | sd {rr.std(ddof=1):4.1f} | 실측과 상관 {np.corrcoef(s, t)[0,1]:+.2f}')
print(f'  (암막 없음 배율 {um_nb:.2f} µm/px. 4개라 상관 |r|>0.95 이어야 우연과 구분. 실측 sd 자체가 {t.std()*1000:.1f}µm)')
