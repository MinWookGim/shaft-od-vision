# 선행 개발자 보고서의 Anchor Tracking 을 흉내 내서 측정 박스를 잘라 둔다
# 보고서: 기준 사진 NO2 20261008_151856.jpg, REFERENCE_BOX (7844, 3468, 8089, 4468), 주변 특징을 template matching -> dx, dy 로 박스 이동
# 원문에 template 영역·탐색 범위가 없어 내가 정함(결과 보기 전 고정):
#   template = 기준 사진의 박스 주변 넓은 영역 (x 7400~8500, y 3200~4700, 칼라·홈 같은 가로 방향 특징 포함)
#   탐색 = 각 사진에서 template 둘레 +-200px, 절반 해상도로 TM_CCOEFF_NORMED (2px 단위)
import os, csv, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
BOX = (7844, 3468, 8089, 4468); TX0, TY0, TX1, TY1 = 7400, 3200, 8500, 4700; S = 200
rd = lambda f: cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(D, f), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
ref = rd('20261008_151856.jpg')
tpl = cv2.resize(ref[TY0:TY1, TX0:TX1], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
out = {}
for i, r in enumerate(rows):
    g = rd(r['파일'])
    win = cv2.resize(g[TY0 - S:TY1 + S, TX0 - S:TX1 + S], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED); _, score, _, loc = cv2.minMaxLoc(res)
    dx, dy = loc[0] * 2 - S, loc[1] * 2 - S
    x0, y0, x1, y1 = BOX
    out[r['파일']] = dict(fixed=g[y0:y1, x0:x1].copy(), track=g[y0 + dy:y1 + dy, x0 + dx:x1 + dx].copy(), dx=dx, dy=dy, score=score)
    print(i, r['시편'], dx, dy, round(score, 3), flush=True)
np.savez_compressed(os.path.join(HERE, 'anchor_crops.npz'), data=np.array([out], dtype=object))
