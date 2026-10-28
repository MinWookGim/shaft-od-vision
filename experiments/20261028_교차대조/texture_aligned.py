# ④ 보강: 질감 상관이 떨어진 것이 '어긋남' 때문인지 '질감 자체가 바뀜' 때문인지 가르기
#   가운데 조각을 위상상관으로 찾은 이동만큼 되돌려 놓고 상관을 다시 계산(맞춘 뒤 상관)
#   비교 기준: 같은 시편 안 바로 앞 사진과의 맞춘 뒤 상관(약 2초 간격, 사진 잡음만 있는 상한)
import glob, os
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
Y, X = (4250, 4500), (6600, 7900)
fs = sorted([f for f in glob.glob(SHARE + '[12]_*/NO*/*.jpg') if '_1533' not in f], key=os.path.basename)
def acorr(a, b):
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=20)
    bb = nd_shift(b, s, order=1)[10:-10, 10:-10]; aa = a[10:-10, 10:-10]
    return np.corrcoef(aa.ravel(), bb.ravel())[0, 1], s
ref = prev = None; rows = []
for f in fs:
    g = np.asarray(Image.open(f).convert('L'), dtype=float)[Y[0]:Y[1], X[0]:X[1]]; g = g - g.mean()
    if ref is None: ref = g
    c_ref, s = acorr(ref, g)
    c_prev = acorr(prev, g)[0] if prev is not None else np.nan
    rows.append(dict(file=os.path.basename(f), stage=f.split('/')[-3][0], no=f.split('/')[-2], c_ref=c_ref, c_prev=c_prev, dy=-s[0], dx=-s[1]))
    prev = g; print(rows[-1]['file'], f'{c_ref:.2f} {c_prev:.2f}', flush=True)
pd.DataFrame(rows).to_csv('texture_aligned.csv', index=False); print('DONE')
