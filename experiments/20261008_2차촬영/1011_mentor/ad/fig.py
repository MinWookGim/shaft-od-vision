# 인공 결함 그린 사진 + 히트맵 (A 겹, 시험 첫 장). 빨간 x = 결함 그린 자리
import os, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumSquare' in f and 'Bold' not in f and 'Extra' not in f: fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['NanumSquare', 'DejaVu Sans']
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', '_out')
fs = sorted(glob.glob(os.path.join(OUT, 'ad_hm_*.npy')))
fig, ax = plt.subplots(2, len(fs), figsize=(3.2 * len(fs), 6.6))
for k, f in enumerate(fs):
    d = np.load(f, allow_pickle=True).item(); lab = os.path.basename(f)[6:-4].replace('_', ' ')
    ax[0, k].imshow(d['img'], cmap='gray', vmin=0, vmax=1); ax[0, k].set_title(lab, fontsize=9)
    ax[1, k].imshow(d['img'], cmap='gray', vmin=0, vmax=1); ax[1, k].imshow(d['hm'], cmap='jet', alpha=.5)
    for a in ax[:, k]: a.set_xticks([]); a.set_yticks([])   # 1차 실행 결과의 결함 자리 표시는 버그라 뺐다
fig.suptitle('인공 결함(위, 위 경계의 혹·패임과 표면 사선)과 이상 히트맵(아래)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'ad_heatmaps.png'), dpi=100)
