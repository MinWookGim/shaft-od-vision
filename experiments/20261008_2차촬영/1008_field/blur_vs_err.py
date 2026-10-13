# 시편별 오차가 위/아래 어느 경계에서 오는지, 경계 모양(번짐)과 관계 있는지
# 경계는 바깥(배경)에서 안쪽으로 들어오며 처음 50% 아래로 내려가는 점. 샤프트 안쪽 반사줄에 안 걸리게
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumSquare' in f and 'Bold' not in f and 'Extra' not in f: fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['NanumSquare', 'DejaVu Sans']
Image.MAX_IMAGE_PIXELS = None
D = '../../field/photos/20261008_원전기계/full_result/'
d = pd.read_csv('_out/full_rows.csv')
first = d.groupby('묶음').head(3)
Y0 = 3600
rows, profs = [], {}
def xcross(a, t):  # a 를 앞에서부터 보며 처음 t 아래로 내려가는 소수점 위치
    i = np.argmax(a < t); return i - 1 + (a[i-1] - t) / (a[i-1] - a[i])
for _, r in first.iterrows():
    g = np.asarray(Image.open(D + r['파일']).convert('L'), dtype=float)
    prof = g[Y0:4320, r['x0']:r['x1']].mean(axis=1)
    bg = r['배경']; dark = np.percentile(prof, 5)
    n = (prof - dark) / (bg - dark)
    top = {t: xcross(n, t) for t in (.9, .5, .1)}               # 위 경계: 위에서 아래로
    nb = n[::-1]; L = len(n) - 1
    bot = {t: L - xcross(nb, t) for t in (.9, .5, .1)}          # 아래 경계: 아래에서 위로
    rows.append(dict(묶음=r['묶음'], 위50=top[.5] + Y0, 아래50=bot[.5] + Y0,
                     위폭_90_10=top[.1] - top[.9], 아래폭_90_10=bot[.9] - bot[.1],
                     위_90_50=top[.5] - top[.9], 위_50_10=top[.1] - top[.5],
                     아래_90_50=bot[.9] - bot[.5], 아래_50_10=bot[.5] - bot[.1]))
    profs.setdefault(r['묶음'], (n, top[.5], bot[.5]))
e = pd.DataFrame(rows).groupby('묶음').mean()
G = pd.read_csv('_out/full_groups.csv', index_col=0)
off = (G.loc[[13, 2, 3], '환산'] - G.loc[[13, 2, 3], '실측']).mean()
e['오차um'] = (G['환산'] - off - G['실측']) * 1000
e['핀중심'] = G['핀중심y']
e['위_핀기준'] = e['위50'] - e['핀중심']; e['아래_핀기준'] = e['아래50'] - e['핀중심']
e['시편'] = G['시편']
print(e.round(1).to_string())
m = e.index != 1   # 옛 핀배치 제외
print()
for c in ['위폭_90_10', '아래폭_90_10', '위_90_50', '위_50_10', '아래_90_50', '아래_50_10', '위_핀기준', '아래_핀기준']:
    print(f'{c:10s} 상관 {np.corrcoef(e.loc[m, c], e.loc[m, "오차um"])[0, 1]:+.2f}  범위 {e.loc[m, c].min():.1f}~{e.loc[m, c].max():.1f}px')
# 그림: 위·아래 경계 밝기 모양을 50% 점에 맞춰 겹쳐 그림
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
y = np.arange(len(n))
for gi, (n, t5, b5) in profs.items():
    lw = 2.2 if G.loc[gi, '역할'] == 'ng' else .9
    ax[0].plot(y - t5, n, lw=lw, label=f"{G.loc[gi,'시편']}({gi})")
    ax[1].plot(b5 - y, n, lw=lw)
for a, t in zip(ax, ['위 경계 (왼쪽=배경 쪽)', '아래 경계 (왼쪽=배경 쪽)']):
    a.set_xlim(-60, 60); a.axhline(.5, c='k', lw=.5); a.axvline(0, c='k', lw=.5); a.set_title(t); a.set_xlabel('50% 점에서 거리 (px, 1px 약 37um)')
ax[0].set_ylabel('밝기 (0=샤프트, 1=배경)'); ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig('_out/edge_profiles.png', dpi=110)
