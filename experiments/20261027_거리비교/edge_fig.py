# 400mm(테스트4) vs 300mm(테스트5) 같은 시편 NO2의 Ø20 단 경계 확대와 밝기 단면 (원본 배열 기준, 화면에서는 실제 방향으로 뒤집음)
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
Image.MAX_IMAGE_PIXELS = None
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
B = '../../ETC/261021/'
sets = [('400mm (테스트4)', B + '테스트4/20261021_144224.jpg', 8420, 36.3, '#1f5fa8'),
        ('300mm (테스트5)', B + '테스트5/20261021_150208.jpg', 8180, 26.7, '#d9463e')]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 1, 1.5]})
for k, (n, f, xc, um, col) in enumerate(sets):
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    # dist_compare.py 와 같은 방식: 창 끝에 닿은 어두운 구간(박스 그림자)은 빼고, 남은 첫 어두운 구간의 시작 = 원본 기준 위 경계
    c = g[3300:4350, xc]
    dk = np.r_[0, (c < 100).astype(int), 0]
    st, en = np.where(np.diff(dk) == 1)[0], np.where(np.diff(dk) == -1)[0] - 1
    keep = (st > 0) & (en < len(c) - 1) & (en - st >= 10)
    top = st[keep][0] + 3300
    # 같은 실제 크기(1.6mm x 1.6mm) 창으로 아래쪽(원본 기준 위) 경계 확대
    h = int(1600 / um)
    crop = g[top - h // 2: top + h // 2, xc - h // 2: xc + h // 2][::-1, ::-1]
    ax[k].imshow(crop, cmap='gray', vmin=0, vmax=255, extent=[0, 1.6, 0, 1.6])
    ax[k].set_title(f'{n}: 1px {um:.1f}µm', fontsize=10); ax[k].set_xlabel('mm')
    # 밝기 단면(가로 21열 평균), µm 단위
    prof = g[top - 60: top + 60, xc - 10: xc + 11].mean(axis=1)
    y = (np.arange(len(prof)) - 60) * um
    ax[2].plot(y, prof, color=col, lw=2, label=n)
ax[2].set_xlim(-1500, 1500); ax[2].set_xlabel('경계에서 거리 (µm)'); ax[2].set_ylabel('밝기')
ax[2].set_title('경계 밝기 단면: 300mm가 약 4배 넓게 번진다', fontsize=10); ax[2].legend(); ax[2].grid(alpha=.3)
fig.suptitle('같은 시편(NO2)·같은 노출(1/40, ISO 40)·같은 조명. 거리만 바꿨을 때 Ø20 단 경계 (10-21 현장 시험)', fontsize=11)
fig.tight_layout(); fig.savefig('fig_dist_edge.png', dpi=120)
