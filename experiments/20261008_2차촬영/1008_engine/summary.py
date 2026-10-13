# engine_rows.csv -> 방법별 · 교정 나누기별 오차표
# 내 방법(50% 밝기 경계) 결과도 같은 표에 넣어 나란히 본다
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
e = pd.read_csv(os.path.join(HERE, '_out', 'engine_rows.csv'))
_t = os.path.join(HERE, '_out', 'engine_rows_tight.csv')      # 박스 안 넓힘 실험(run_tight.py)
if os.path.exists(_t): e = pd.concat([e, pd.read_csv(_t)], ignore_index=True)
m = pd.read_csv(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'))
m = m.rename(columns={'샤프트sd': '샤프트sd_'})[['파일', '묶음', '시편', '역할', '실측mm', '샤프트px', '핀px']]
m['방법'] = 'M_내방법_50%경계'
d = pd.concat([e, m], ignore_index=True)
d['환산'] = d['샤프트px'] / d['핀px'] * 10            # 핀 Ø10 비율로 mm

SPLITS = {'계획(1,2,3)': ([1, 2, 3], [4, 5, 6]), '핀같은(13,2,3)': ([13, 2, 3], [4, 5, 6]),
          '선행(4,5,6)': ([4, 5, 6], [1, 2, 3, 13])}
RAND, NG = [7, 8, 9, 10], [11, 12]
lines = []
for name, dd in d.groupby('방법'):
    g = dd.groupby('묶음').agg(시편=('시편', 'first'), 실측=('실측mm', 'first'), 환산=('환산', 'mean'),
                             장sd=('환산', 'std'), 실패=('샤프트px', lambda s: s.isna().sum()))
    rep = g['장sd'].median() * 1000
    lines.append(f'\n=== {name}  (같은 시편 10장 흔들림 중앙값 {rep:.1f}um, 측정 실패 {int(g["실패"].sum())}장)')
    for sname, (cal, val) in SPLITS.items():
        off = (g.loc[cal, '환산'] - g.loc[cal, '실측']).mean()
        err = (g['환산'] - off - g['실측']) * 1000
        s = lambda ix: f'MAE {err[ix].abs().mean():5.1f} 최대 {err[ix].abs().max():5.1f}'
        lines.append(f'  교정 {sname:12s} 검증{val} {s(val)} | 랜덤7~10 {s(RAND)} [{", ".join(f"{v:+.0f}" for v in err[RAND])}]'
                     f' | NG {err[NG[0]]:+.0f}, {err[NG[1]]:+.0f}')
txt = '\n'.join(lines); print(txt)
open(os.path.join(HERE, '_out', 'summary.txt'), 'w').write(txt)
