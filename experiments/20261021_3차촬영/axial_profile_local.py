# 3번 재실행: axial_profile.py 는 기존 tb(위 경계를 바닥에서 잡는 결함)를 써서, 국소 50% 방식(local_edges.py 의 height)으로 다시
# 출력: axial_profile_local.npz (files, xs, d=지름 px, top, bot)
from pathlib import Path
import numpy as np
from PIL import Image
exec(open('local_edges.py').read().split('ps = sorted')[0])     # refine, height, Y0, Y1, NO1_OFF 가져오기 (본문은 실행 안 함)
XS = np.arange(7955, 8401, 2)
files, D, T, B = [], [], [], []
for p in sorted(SHARE.glob('[12]_*/NO*/*.jpg'), key=lambda q: q.name):
    dx, dy = NO1_OFF if '_1533' in p.name else (0, 0)
    g = np.asarray(Image.open(p).convert('L'), dtype=float)
    r = np.array([height(g, x - dx, dy) for x in XS])
    files.append(p.name); D.append(r[:, 0]); T.append(r[:, 1] + Y0); B.append(r[:, 2] + Y0)
np.savez('axial_profile_local.npz', files=np.array(files), xs=XS, d=np.array(D), top=np.array(T), bot=np.array(B))
print('done', len(files))
