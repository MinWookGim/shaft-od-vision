# 10-26 거칠기 12개(업체 10-24 측정, roughness_20261024.csv)와 시편별 치우침(3차 분석정리 5절, 시간 보정 후) 대조
# 실행: 이 폴더에서 python rough_vs_bias.py > rough_vs_bias.txt
import numpy as np
from scipy import stats
names = ['NO1','NO2','NO3','NO4','NO5','NO6','NO7','NO8','NO9','NO10','NG1','NG2']
Ra = [1.123,1.587,2.173,1.369,1.352,1.731,2.035,1.561,1.231,2.144,1.641,1.215]
Rz = [4.921,12.099,14.381,9.449,10.882,9.769,9.447,6.785,5.965,16.375,9.156,7.976]
bias = [12,1,-13,-7,-9,-1,0,5,-16,-8,-22,-13]      # 예측-실측 µm (분석정리 5절)
Ra, Rz, bias = map(np.array, (Ra, Rz, bias))
rng = np.random.default_rng(0)
def perm_p(x, y, n=20000):
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])
    return r0, (np.abs(rs) >= abs(r0)).mean()
for lab, m in [('12개', np.ones(12, bool)), ('NO1 제외 11개', np.array([n != 'NO1' for n in names]))]:
    print('==', lab)
    for xn, x in [('Ra', Ra), ('Rz', Rz)]:
        for yn, y in [('치우침', bias), ('|치우침|', np.abs(bias))]:
            r, p = perm_p(x[m], y[m]); rho = stats.spearmanr(x[m], y[m])[0]
            print(f'{xn} vs {yn}: r={r:+.2f} (순위 {rho:+.2f}) 순열 p={p:.2f}')
print('Ra-Rz r', np.corrcoef(Ra, Rz)[0,1].round(2))
# 예전 3개(같은 업체, 이전 측정)와 비교
old = {'NO7': 8.8, 'NO10': 8.1, 'NG1': 6.8}
for k, v in old.items(): print(k, '예전 Rz', v, '-> 10-24 Rz', Rz[names.index(k)])

# 그림: Ra·Rz vs 시편별 치우침
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for a, x, xn in [(ax[0], Rz, 'Rz (µm)'), (ax[1], Ra, 'Ra (µm)')]:
    for n, xi, yi in zip(names, x, bias):
        a.plot(xi, yi, 'o', color='#d32f2f' if n.startswith('NG') else '#1f5fa8', ms=7)
        a.annotate(n, (xi, yi), xytext=(4, 4), textcoords='offset points', fontsize=8)
    r = np.corrcoef(x, bias)[0, 1]
    a.axhline(0, color='k', lw=.6); a.axhspan(-10, 10, color='#dff0d8', alpha=.5)
    a.set_xlabel(xn); a.set_title(f'{xn.split()[0]} vs 시편별 치우침: 상관 {r:+.2f}', fontsize=10); a.grid(alpha=.3)
ax[0].set_ylabel('시편별 치우침 (µm, 예측 - 실측, 시간 보정 후)')
fig.suptitle('표면 거칠기(10-24 업체 측정 12개)와 시편별 치우침: 관계가 보이지 않는다 (12개면 |상관| 0.58 이상이어야 우연과 구분)', fontsize=10.5)
fig.tight_layout(); fig.savefig('../../docs/images/20261026_rough/fig_rough_vs_bias.png', dpi=130)
