# ④ 배경(역광판) 질감이 사진 안에서 움직였나: 샤프트 아래 밝은 띠에서 왼쪽·가운데·오른쪽 세 조각을
# 첫 사진(NO2 본시험 첫 장)과 위상상관으로 맞춰 x·y 이동(px)을 1/20px 단위로 측정
# 세 조각이 똑같이 움직이면 '평행 이동', 바깥 조각이 서로 반대로 움직이면 '배율 변화'
import glob, os
import numpy as np, pandas as pd
from PIL import Image
from skimage.registration import phase_cross_correlation
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
Y = (4250, 4500)
PATCH = {'L': (5200, 6400), 'M': (6600, 7900), 'R': (8500, 10000)}
fs = sorted([f for f in glob.glob(SHARE + '[12]_*/NO*/*.jpg') if '_1533' not in f], key=os.path.basename)
ref = None; rows = []
for f in fs:
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    p = {k: g[Y[0]:Y[1], a:b] for k, (a, b) in PATCH.items()}
    if ref is None: ref = p
    r = dict(file=os.path.basename(f), stage=f.split('/')[-3][0], no=f.split('/')[-2])
    for k in PATCH:
        s, err, _ = phase_cross_correlation(ref[k] - ref[k].mean(), p[k] - p[k].mean(), upsample_factor=20)
        r[f'{k}_dy'], r[f'{k}_dx'] = -s[0], -s[1]
        r[f'{k}_corr'] = np.corrcoef(ref[k].ravel(), p[k].ravel())[0, 1]
        r[f'{k}_mean'] = p[k].mean()
    rows.append(r); print(r['file'], f"{r['M_dx']:+.2f} {r['M_corr']:.2f}", flush=True)
pd.DataFrame(rows).to_csv('texture.csv', index=False)
print('DONE')
