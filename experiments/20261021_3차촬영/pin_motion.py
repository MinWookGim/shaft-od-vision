# 핀이 건드려졌나(계단) vs 서서히 흐름(폰·배율) 가르기. 암막 150장 전부(NO1 제외)
# 핀: 위·아래 경계 y, 중심 y, 높이 / 핀 오른쪽 끝면 x(축 방향 위치) / V블록 오른쪽 경계 x
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from pos_scan_lib import cross
Image.MAX_IMAGE_PIXELS=None
SH=Path('../../field/photos/20261021_원전기계/3차_261021_분류')
fs=sorted([p for p in SH.glob('[12]_*/NO*/*.jpg') if '_1533' not in p.name], key=lambda p:p.name)
rows=[]
for p in fs:
    g=np.asarray(Image.open(p).convert('L'),dtype=float)
    tb=[]
    for x in range(5400,5800,10):
        col=g[3000:4600,x]; n=len(col)//2
        tb.append((3000+cross(col[:n]), 3000+(n-1-cross(col[n:][::-1]))+n))
    tb=np.array(tb); T,B=np.nanmedian(tb[:,0]),np.nanmedian(tb[:,1])
    # 핀 끝면 x: 핀 가운데 높이 행들에서 어두움->밝음 (오른쪽 끝, x 5800~6100)
    ends=[5800+cross(g[y,5800:6100]) for y in range(int(T)+150,int(B)-150,25)]
    R=[10000+cross(g[y,10000:10700]) for y in range(4200,4400,20)]
    rows.append(dict(file=p.name,stage=p.parts[-3][0],no=int(p.parts[-2][2:]),T=T,B=B,C=(T+B)/2,H=B-T,end=np.nanmedian(ends),vR=np.median(R)))
d=pd.DataFrame(rows); d.to_csv('pin_motion.csv',index=False)
d['grp']=d.stage+'-'+d.no.astype(str)
g=d.groupby('grp',sort=False)
print('[놓임 안(약 20초) 변화: 첫 장 -> 끝 장, 본시험만]')
for k,s in g:
    if s.stage.iloc[0]!='1': continue
    print(f'  {k:6s} 핀높이 {s.H.iloc[0]:.2f}->{s.H.iloc[-1]:.2f} ({s.H.iloc[-1]-s.H.iloc[0]:+.2f})  중심y {s.C.iloc[-1]-s.C.iloc[0]:+.2f}  끝면x {s.end.iloc[-1]-s.end.iloc[0]:+.2f}  V블록 {s.vR.iloc[-1]-s.vR.iloc[0]:+.2f}')
a=g.agg(t=('file','first'),H=('H','mean'),Hsd=('H','std'),C=('C','mean'),T=('T','mean'),B=('B','mean'),end=('end','mean'),endsd=('end','std'),vR=('vR','mean')).reset_index()
for c in ['H','C','T','B','end','vR']: a['d'+c]=a[c].diff()
print('\n[시편 바뀔 때 계단: 앞 놓임 평균 대비]')
print(a[['grp','t','H','dH','C','dC','dT','dB','end','dend','vR','dvR']].round(2).to_string(index=False))
within=d[d.stage=='1'].groupby('grp').apply(lambda s:s.H.iloc[-1]-s.H.iloc[0]).sum()
print(f'\n본시험 놓임 안 변화 합 {within:+.2f}px  vs  전체 변화(NO2 -> NO12) {a.H.iloc[10]-a.H.iloc[0]:+.2f}px')
