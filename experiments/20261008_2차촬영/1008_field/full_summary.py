# full_rows.csv -> 묶음별 요약 + 핀 비율 환산 + 교정 시편으로 치우침 빼기
import pandas as pd, numpy as np
d = pd.read_csv('_out/full_rows.csv')
d['r'] = d['샤프트px'] / d['핀px'] * 10          # 핀 Ø10 기준 환산 mm
g = d.groupby('묶음').agg(시편=('시편','first'), 역할=('역할','first'), 실측=('실측mm','first'), 초점=('초점','first'), 핀=('핀배치','first'),
    장=('파일','size'), 샤프트px=('샤프트px','mean'), 샤프트px_sd=('샤프트px','std'), 열sd=('샤프트sd','median'),
    핀px=('핀px','mean'), 핀px_sd=('핀px','std'), 중심y=('샤프트중심y','mean'), 핀중심y=('핀중심y','mean'),
    흐림위=('흐림위','median'), 흐림아래=('흐림아래','median'), 배경=('배경','mean'), 포화=('포화%','max'),
    환산=('r','mean'), 환산sd=('r','std'))
g['핀만_오차um'] = (g['환산'] - g['실측']) * 1000
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(g.round(3).to_string())
for name, cal in [('계획대로 NO1~3(묶음1,2,3)', [1,2,3]), ('같은 핀배치 NO1재촬영+NO2,3(묶음13,2,3)', [13,2,3])]:
    off = (g.loc[cal,'환산'] - g.loc[cal,'실측']).mean()
    e = (g['환산'] - off - g['실측']) * 1000
    print(f'\n[{name}] 치우침 {off*1000:+.1f}um')
    for gi in g.index: print(f"  {gi:2d} {g.loc[gi,'시편']:5s} {g.loc[gi,'역할']:6s} 실측 {g.loc[gi,'실측']:.3f} 추정 {g.loc[gi,'환산']-off:.4f} 오차 {e[gi]:+6.1f}um  장당sd {g.loc[gi,'환산sd']*1000:4.1f}um")
    for role in ['val','rand','ng']:
        m = g['역할']==role; print(f'  {role}: MAE {e[m].abs().mean():.1f}um  최대 {e[m].abs().max():.1f}um')
g.to_csv('_out/full_groups.csv', encoding='utf-8-sig')
