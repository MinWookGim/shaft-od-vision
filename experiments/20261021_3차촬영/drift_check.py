# 핀 px 가 시간에 따라 줄어드는 원인 가르기: 배경 밝기 vs 배율(위치) vs 번짐
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from pos_scan_lib import cross
Image.MAX_IMAGE_PIXELS=None
SH=Path('../../field/photos/20261021_원전기계/3차_261021_분류')
fs=sorted([p for p in SH.glob('[12]_*/NO*/*.jpg') if '_1533' not in p.name], key=lambda p:p.name)
fs=fs[::3]
def tb(g,x):
    col=g[3000:4600,x]; n=len(col)//2
    t=cross(col[:n]); b=cross(col[n:][::-1]); return 3000+t, 3000+(n-1-b)+n
def edge_w(g,x,y,up):
    # 10-90% 전이폭 (위 경계 up=True)
    seg=g[int(y)-40:int(y)+41,x].astype(float)
    if not up: seg=seg[::-1]
    a,b=seg[:10].mean(),seg[-10:].mean(); lo,hi=b+0.1*(a-b),b+0.9*(a-b)
    i90=np.argmax(seg<hi); i10=np.argmax(seg<lo); return i10-i90
rows=[]
for p in fs:
    g=np.asarray(Image.open(p).convert('L'),dtype=float)
    pt=[tb(g,x) for x in range(5400,5800,40)]; st=[tb(g,x) for x in range(8040,8320,40)]
    pT,pB=np.median([a for a,b in pt]),np.median([b for a,b in pt]); sT,sB=np.median([a for a,b in st]),np.median([b for a,b in st])
    rows.append(dict(file=p.name, stage=p.parts[-3][:1], bg_up=g[3150:3250,8050:8300].mean(), bg_dn=g[4250:4350,8050:8300].mean(),
        bg_pin_up=g[3200:3300,5400:5800].mean(), pinT=pT,pinB=pB,pin=pB-pT,shT=sT,shB=sB,sh=sB-sT,
        ew_pin_up=np.median([edge_w(g,x,pT,True) for x in range(5400,5800,40)]), ew_pin_dn=np.median([edge_w(g,x,pB,False) for x in range(5400,5800,40)])))
d=pd.DataFrame(rows); pd.set_option('display.width',250)
print(d.round(2).to_string(index=False))
for c in ['bg_up','bg_dn','bg_pin_up','pinT','pinB','shT','shB','ew_pin_up','ew_pin_dn']:
    print(c, 'corr with pin px: %.2f'%np.corrcoef(d[c],d.pin)[0,1])
