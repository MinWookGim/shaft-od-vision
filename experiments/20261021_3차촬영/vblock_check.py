# 고정 물체 두 번째 기준: 두 V블록 안쪽 세로 경계 사이 거리(약 5000px). 배율이 0.4% 바뀌면 약 20px 움직여야 함
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from pos_scan_lib import cross
Image.MAX_IMAGE_PIXELS=None
SH=Path('../../field/photos/20261021_원전기계/3차_261021_분류')
fs=sorted([p for p in SH.glob('[12]_*/NO*/*.jpg') if '_1533' not in p.name], key=lambda p:p.name)[::3]
rows=[]
for p in fs:
    g=np.asarray(Image.open(p).convert('L'),dtype=float)
    L=[];R=[]
    for y in range(4200,4400,20):
        row=g[y]
        L.append(4900+cross(row[4900:5500])); R.append(10000+cross(row[10000:10700]))
    rows.append(dict(file=p.name,L=np.median(L),R=np.median(R),W=np.median(R)-np.median(L)))
d=pd.DataFrame(rows); v=pd.read_csv('v3_d20.csv')[['file','pin_px']]; d=d.merge(v,on='file')
print(d.round(2).to_string(index=False))
print('corr W vs pin %.2f'%np.corrcoef(d.W,d.pin_px)[0,1], ' W range %.1f'%(d.W.max()-d.W.min()))
