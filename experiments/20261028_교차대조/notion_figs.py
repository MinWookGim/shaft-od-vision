# 노션 공유용 그림 5장: 교정식 비교 / 시편 공통 튀는 열 / 대표값 31열 vs 21열 / 재장착 반복성 / 흐림+감마 개념도
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
B, O, A, Yl = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#2b2b2a', '#8a8a85', '#e6e5e0'
OUT = 'notion_figs/'
def style(ax, grid='x'):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK, labelsize=9.5)
def title(fig, t, sub=None):
    fig.text(0.02, 0.97, t, fontsize=13, color=INK, weight='bold', va='top')
    if sub: fig.text(0.02, 0.915, sub, fontsize=9.5, color=MUTED, va='top')

# 1. 교정식 비교 (점 = 경계 원천)
r = pd.read_csv('cross_calib.csv'); r = r[r.사진 == 'all']
order = ['2 전체 선형(기울기 자유)', '4 핀 비율 선형(기울기 자유)', '3 물리 배율(V1)', '3a 물리 + 핀 px 1:1 더함(내 보정)', '5 핀 비율 x16 고정']
lab = ['전체 선형 (기울기 자유)', '핀 비율 선형 (기울기 자유)', '물리 배율 고정', '물리 배율 + 시간 보정', '핀 비율 x16 (배율 고정)']
src = [('팀원 cross50 (약 110열 강건 직선)', '이다겸 경계 (약 110열)', B, 'o'), ('내 국소50% 32열 중앙값 (3차 분석정리)', '김민욱 경계 32열', O, 's'),
       ('내 국소50% 31열 평균 (대표값 비교)', '김민욱 경계 31열 평균', A, '^'), ('내 국소50% 21열 평균 (튀는 열 뺌)', '김민욱 경계 21열 평균', Yl, 'D')]
fig, ax = plt.subplots(figsize=(9.5, 4.6)); fig.subplots_adjust(left=0.27, top=0.74, bottom=0.14, right=0.97)
ax.axhspan(-0.5, 1.5, color='#eef4fc', zorder=0)
for k, (s, l, c, m) in enumerate(src):
    v = [r[(r.경계 == s) & (r.방식 == o)].MAE.iloc[0] for o in order]
    ax.scatter(v, np.arange(5) + (k - 1.5) * 0.12, color=c, marker=m, s=46, label=l, zorder=3, edgecolor='white', lw=0.8)
ax.axvline(6.64, color=INK, lw=1.2, ls='--'); ax.text(6.70, -0.55, '평균 부르기 6.64', fontsize=9, color=INK)
ax.set_yticks(range(5), lab); ax.invert_yaxis(); ax.set_xlim(3, 10); ax.set_xlabel('시편 하나씩 빼고 맞히기 MAE (µm, 정상 9개)', color=INK)
style(ax); ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=4, fontsize=8.5, frameon=False, handletextpad=0.2, columnspacing=1.0)
ax.text(3.1, 0.5, '기울기를 데이터로 맞춤\n(물리 배율의 0.35 ∼ 0.6배)', fontsize=8.5, color=B, va='center')
title(fig, '최선 MAE 3.99는 경계보다 "기울기를 자유롭게 맞춘 교정식" 몫이 크다', '같은 3차 사진, 경계 원천 4가지 x 교정식 5가지. 기울기를 자유롭게 맞추면 어느 경계든 4 ∼ 5.6µm')
fig.savefig(OUT + 'fig1_calib.png', dpi=140); plt.close()

# 2. 시편 공통으로 튀는 열 (시편 11개 첫 사진, 1px 열)
a = np.load('fine_cols.npy'); XF = np.arange(8000, 8341)
fig, axs = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True); fig.subplots_adjust(left=0.1, top=0.82, bottom=0.11, right=0.97, hspace=0.15)
XS = np.linspace(8030, 8320, 31).round().astype(int)
for k, (ax, name) in enumerate(zip(axs, ['위 경계', '아래 경계'])):
    R = []
    for y in a[:, :, k]:
        mk = ~np.isnan(y); p = np.polyfit(XF[mk] - 8170, y[mk], 3); R.append((y - np.polyval(p, XF - 8170)) * 26.7)
    R = np.array(R)
    for rr in R: ax.plot(XF, rr, color=MUTED, lw=0.5, alpha=0.35)
    ax.plot(XF, np.nanmedian(R, 0), color=B, lw=1.8)
    ax.axvspan(8069, 8262, color='#eef4fc', zorder=0)
    ax.scatter(XS, np.full(31, 95), marker='|', color=O, s=40)
    ax.set_ylim(-120, 110); ax.set_ylabel(f'{name}\n추세 뺀 편차 (µm)', color=INK, fontsize=9.5); style(ax, 'y')
axs[0].annotate('x 8279 ∼ 8282: 11개 시편 모두 -70 ∼ -84µm', xy=(8280, -84), xytext=(8130, -108), fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=INK, lw=0.8))
axs[0].text(8072, 60, '튀는 열을 뺀 창 (8069 ∼ 8262)', fontsize=9, color=B)
axs[0].text(8000, 100, '| = 처음 쓴 31개 열', fontsize=8.5, color=O, va='bottom')
axs[1].set_xlabel('원본 사진 x 좌표 (px, Ø20 단)', color=INK)
title(fig, '사진에는 시편과 상관없이 같은 자리에서 튀는 열이 있다', '회색 = 시편 11개 각각(첫 사진), 파랑 = 시편 중앙값. 같은 자리라서 시편 표면이 아니라 사진 쪽 요인')
fig.savefig(OUT + 'fig2_cols.png', dpi=140); plt.close()

# 3. 대표값 31열 vs 21열 (물리 배율)
q = pd.read_csv('repr_clean.csv'); q = q[q.방식 == '3 물리']
cand = ['Median', 'Mean', 'P90', 'P95', '상위3', '상위5', 'Max', 'envelope']
fig, ax = plt.subplots(figsize=(9.5, 4.6)); fig.subplots_adjust(left=0.14, top=0.78, bottom=0.13, right=0.97)
for i, c in enumerate(cand):
    v31 = q[(q.대표값 == c) & (q.창 == '31열')].MAE.iloc[0]; v21 = q[(q.대표값 == c) & (q.창 == '21열')].MAE.iloc[0]
    ax.plot([v31, v21], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter(v31, i, color=MUTED, s=44, zorder=2, label='31개 열 (튀는 열 포함)' if i == 0 else None)
    ax.scatter(v21, i, color=B, s=50, zorder=3, label='21개 열 (튀는 열 뺌)' if i == 0 else None)
    ax.text(v21, i - 0.32, f'{v21:.1f}', fontsize=9, color=B, ha='center')
ax.axvline(6.64, color=INK, lw=1.2, ls='--'); ax.text(6.8, -0.6, '평균 부르기 6.64', fontsize=9, color=INK)
ax.set_yticks(range(8), cand); ax.invert_yaxis(); ax.set_xlim(0, 40); ax.set_xlabel('MAE (µm, 물리 배율, 정상 9개 하나씩 빼기)', color=INK)
style(ax); ax.legend(loc='center right', fontsize=9, frameon=False)
title(fig, '대표값: 가운데 값(Mean·Median)이 가장 가깝고, 끝값은 튀는 열에 크게 흔들린다', 'Max·envelope는 튀는 열을 빼면 2 ∼ 3배 좋아지지만 여전히 가운데 값보다 나쁘다. 어느 후보도 평균 부르기는 못 이김')
fig.savefig(OUT + 'fig3_repr.png', dpi=140); plt.close()

# 4. 재장착 반복성: 사진 잡음 vs 놓임 몫
m = pd.read_csv('remount_cross.csv')
names = ['이다겸 경계\n(약 110열)', '김민욱 경계\n32열 중앙값', '김민욱 경계\n31열 평균', '김민욱 경계\n21열 평균']
fig, ax = plt.subplots(figsize=(9.5, 4.4)); fig.subplots_adjust(left=0.08, top=0.8, bottom=0.17, right=0.97)
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, m.사진사이_내정의, w, color=B, label='사진 한 장의 경계 잡음 (같은 놓임 sd)')
b2 = ax.bar(x + w / 2 + 0.01, m.놓임몫, w, color=O, label='다시 놓을 때 생기는 몫')
for bb in list(b1) + list(b2): ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.15, f'{bb.get_height():.1f}', ha='center', fontsize=9, color=INK)
for i in range(4): ax.text(i, -2.3, f'%GRR {m.GRR_내정의[i]}', ha='center', fontsize=9, color=MUTED)
ax.set_xticks(x, names); ax.set_ylabel('µm', color=INK); ax.set_ylim(0, 10.5); style(ax, 'y'); ax.legend(fontsize=9, frameon=False, loc='upper right')
title(fig, '재장착 반복성 차이(5.2 vs 11.6µm)는 사진 한 장의 경계 잡음 차이다', '열을 많이 평균할수록 잡음이 준다. 다시 놓는 몫은 2.7 ∼ 3.8µm로 비슷. 가장 좋은 경계도 %GRR 128 ∼ 155%(기준 30%)')
fig.savefig(OUT + 'fig4_remount.png', dpi=140); plt.close()

# 5. 흐림 + 감마 개념도
xx = np.linspace(-6, 6, 600)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2)); fig.subplots_adjust(left=0.07, top=0.78, bottom=0.15, right=0.98, wspace=0.25)
for ax, (gam, ttl) in zip(axs, [(1.0, '빛 그대로 (선형, RAW·산업용 카메라)'), (2.2, '폰 JPEG (밝기 곡선 = 감마 2.2)')]):
    for sig, c, l in [(1.0, B, '선명'), (1.7, O, '흐림')]:
        lin = norm.cdf(xx / sig); code = lin ** (1 / gam)
        ax.plot(xx, code, color=c, lw=2, label=l)
        xc = xx[np.argmin(abs(code - 0.5))]; ax.plot([xc, xc], [0, 0.5], color=c, lw=1, ls=':'); ax.scatter([xc], [0.5], color=c, s=25, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls='--'); ax.axvline(0, color=GRID, lw=1)
    ax.text(2.2, 0.44, '밝기값 50%\n= 경계로 잡는 곳', fontsize=8.5, color=MUTED)
    ax.text(-5.8, 0.06, '물체 (어두움)', fontsize=9, color=INK); ax.text(2.6, 0.93, '배경 (밝음)', fontsize=9, color=INK)
    ax.set_title(ttl, fontsize=10, color=INK); ax.set_xlabel('경계에서 거리 (px)', color=INK); style(ax, 'y'); ax.set_ylim(0, 1.05)
axs[0].legend(fontsize=9, frameon=False, loc='center right')
axs[1].annotate('흐려지면 경계가\n물체 안쪽으로 밀림\n-> 지름이 작게 잡힘', xy=(-0.9, 0.5), xytext=(-5.6, 0.72), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
title(fig, '경계가 흐려지면 왜 작게 잡히나: 폰 사진의 밝기 곡선', '선형이면 흐려져도 50% 지점이 그대로다. 감마를 거친 폰 사진은 흐려질수록 물체 안쪽으로 밀린다(개념도, 가우스 경계 가정)')
fig.savefig(OUT + 'fig5_gamma.png', dpi=140); plt.close()
print('ok')
