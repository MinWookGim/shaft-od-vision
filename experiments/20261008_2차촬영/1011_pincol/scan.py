# 핀게이지를 열마다 측정한다 (10-08 130장 전체)
# 열 x = 9450~9900 을 4px 간격으로, 세로 3700~4250 창에서 위·아래 경계(50% 밝기, 10-08 분석과 같은 규칙)
# 저장: 사진마다 열별 위 경계 y, 아래 경계 y
import os, sys, csv, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '1008_field')); from t1_analyze import edges
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
XS = np.arange(9450, 9901, 4); Y0, Y1 = 3700, 4250
TOP = np.full((len(rows), len(XS)), np.nan); BOT = TOP.copy()
for i, r in enumerate(rows):
    g = np.asarray(Image.open(os.path.join(D, r['파일'])).convert('L'), dtype=float)
    bg = g[4500:4800, 6800:7600].mean()
    for j, x in enumerate(XS):
        col = g[Y0:Y1, x - 2:x + 3].mean(1)           # 5열 평균
        thr = (bg + np.percentile(col, 5)) / 2
        try: t, b = edges(col, thr)                    # 창 끝까지 어두우면(V블록·핀 끝 밖) 경계 없음
        except IndexError: continue
        if 0 < t and b < Y1 - Y0 - 1: TOP[i, j], BOT[i, j] = t + Y0, b + Y0
    print(i, r['파일'], flush=True)
np.savez(os.path.join(HERE, '_out', 'pin_scan.npz'), XS=XS, TOP=TOP, BOT=BOT, files=[r['파일'] for r in rows])
print('saved')
