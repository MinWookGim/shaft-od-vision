# 후보정(선명화·잡음제거) 가설 vs 기하(배율·초점) 가설 가르기
# 후보정이면: 위·아래 경계가 같은 px 씩 안쪽으로(대칭), 경계 번짐폭이 같이 변함, 고정된 세로 경계 위치는 거의 안 움직임
# 배율이면: 경계가 화면 중심(y 3766, x 8160)에서 거리에 비례해 움직임
import pandas as pd, numpy as np
d=pd.read_csv('pin_motion.csv'); d=d[d.stage==1]
m=d.groupby('no')[['T','B','H','vR','end']].mean()
a,b=m.iloc[0],m.iloc[-1]
print('NO2 -> NO12 변화(px): 핀 위경계 %+.2f  아래경계 %+.2f  높이 %+.2f  V블록 세로경계 %+.2f  핀 끝면 %+.2f'%tuple(b-a))
k=(b.H-a.H)/a.H
print('핀 높이 비율 변화 %.3f%%' % (100*k))
print('배율이라면 예상: 위경계 %+.2f 아래경계 %+.2f V블록 %+.2f' % (-k*(3766-a['T']), k*(a['B']-3766), k*(a.vR-8160)))
print('후보정(대칭 침식)이라면 예상: 위경계 %+.2f 아래경계 %+.2f V블록 약 0' % (-(b.H-a.H)/2, (b.H-a.H)/2))
