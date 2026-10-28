# ② 창 점검: 본시험 시편마다 첫 사진 1장에서 Ø20 단 전 구간(8000~8340)을 1px 열마다 국소 50% 경계로 측정
# 목적: 내 31개 열에서 튀던 열(8049·8281·8320)과 팀원 E12의 끝열 스파이크(x 약 8265)가 같은 곳인지 본다
# 경계 측정은 1027_repr/repr_cols.py 와 같다(열마다 3px 평균 단면, 창 끝에 닿은 어두운 띠 제외)
import glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '../1027_repr')
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
src = open('../1027_repr/repr_cols.py').read().split("fs = sorted")[0]
exec(src.replace("Image.MAX_IMAGE_PIXELS = None", ""))
XF = np.arange(8000, 8341)
out = []
for d in sorted(glob.glob(SHARE + '1_*/NO*')):
    if d.endswith('NO01'): continue
    f = sorted(glob.glob(d + '/*.jpg'))[0]
    g = np.asarray(Image.open(f).convert('L'), dtype=float)
    r = np.array([edges(g, x, 0, 0) for x in XF])
    out.append(r); print(os.path.basename(d), os.path.basename(f), flush=True)
np.save('fine_cols.npy', np.array(out))
