# 10-08 130장에서 선행 엔진 측정 박스(샤프트, 핀)만 잘라 저장한다. 전처리 변형을 여러 번 돌릴 때 2억 화소 사진을 다시 안 읽으려고
import os, sys, csv, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, '..', '1008_engine'); sys.path.insert(0, E)
import run_engine as R                      # 박스 정의(Y1,Y2,PIN)와 사진 목록을 그대로 쓴다
out = {}
for i, rw in enumerate(R.rows):
    img = cv2.imdecode(np.fromfile(os.path.join(R.D, rw['파일']), np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, x1 = int(rw['x0']), int(rw['x1'])
    out[rw['파일']] = dict(shaft=g[R.Y1:R.Y2, x0:x1].copy(), pin=g[R.PIN[1]:R.PIN[3], R.PIN[0]:R.PIN[2]].copy())
    print(i, flush=True)
np.savez_compressed(os.path.join(HERE, 'crops.npz'), data=np.array([out], dtype=object))
