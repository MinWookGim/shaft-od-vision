# 130장에서 경계 근처 밝기만 잘라 저장한다. 방법을 바꿀 때마다 2억 화소 사진을 다시 읽지 않으려고
# 샤프트: 베어링부 열(x0~x1, 사진마다 다름) x 세로 3400~4350 (위 가림막 경계 ~3465, 샤프트 ~3678~4225 포함)
# 핀: 열 9620~9840 x 세로 3750~4150
import os, csv, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
SY = (3400, 4350); PY = (3750, 4150); PX = (9620, 9840); BG = (4500, 4800, 6800, 7600)
out = {}
for i, r in enumerate(rows):
    g = np.asarray(Image.open(os.path.join(D, r['파일'])).convert('L'))
    x0, x1 = int(r['x0']), int(r['x1'])
    out[r['파일']] = dict(shaft=g[SY[0]:SY[1], x0:x1].copy(), x0=x0, pin=g[PY[0]:PY[1], PX[0]:PX[1]].copy(),
                         bg=float(g[BG[0]:BG[1], BG[2]:BG[3]].mean()))
    print(i, r['파일'], flush=True)
np.savez_compressed(os.path.join(HERE, '_out', 'cache.npz'), data=np.array([out], dtype=object),
                    meta=np.array([dict(SY=SY, PY=PY, PX=PX)], dtype=object))
print('saved')
