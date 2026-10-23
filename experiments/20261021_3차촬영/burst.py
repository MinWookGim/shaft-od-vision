# 놓임(연속 10장) 안에서 사진 순번에 따른 기울기를 모아 본다. 사람 손이 안 닿는 20초
import pandas as pd, numpy as np
d=pd.read_csv('pin_motion.csv'); d=d[d.stage==1].copy()
d['i']=d.groupby('no').cumcount()
v=pd.read_csv('v3_d20.csv')[['file','p50_px']]; d=d.merge(v,on='file')
for c in ['H','vR','C','end','p50_px']:
    dm=d[c]-d.groupby('no')[c].transform('mean'); im=d.i-d.groupby('no').i.transform('mean')
    b=(dm*im).sum()/(im**2).sum(); res=dm-b*im
    se=np.sqrt((res**2).sum()/(len(d)-d.no.nunique()-1)/(im**2).sum())
    print(f'{c:7s} 놓임 안 기울기 {b:+.4f} px/장 (±{se:.4f}), 10장이면 {9*b:+.2f}px')
# 놓임 평균의 시간 흐름 (사진 첫 장 기준)
m=d.groupby('no').agg(H=('H','mean'),vR=('vR','mean'),H0=('H','first'),H9=('H','last'))
m['next_first-this_last']=m.H0.shift(-1)-m.H9
print(m.round(2).to_string())
