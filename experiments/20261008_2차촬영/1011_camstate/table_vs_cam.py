# 움직인 게 카메라인가 테이블인가 (2026-10-11)
# 삼각대는 바닥(테이블과 따로), 셔터는 블루투스 리모컨(팀장 확인)
#   테이블 밖: 왼쪽 뒤 기계판(구멍 판) — 카메라가 움직이면 이것도 움직이고, 테이블만 움직이면 그대로
#   테이블 위: 왼쪽 받침대 아랫부분, 오른쪽 받침대 아랫부분, 자 눈금
# 샤프트 막대가 지나는 높이(y 3600~4300)는 뺀다. 묶음마다 첫 장 vs 묶음 3 첫 장, 위상상관
import os, csv, numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
ST = list(csv.DictReader(open(os.path.join(HERE, '_out', 'state.csv'), encoding='utf-8-sig')))
first = {}; last = {}
for r in ST: first.setdefault(int(r['묶음']), r['파일']); last[int(r['묶음'])] = r['파일']
REG = {'기계판(테이블 밖) 위': (850, 2100, 700, 3400), '기계판(테이블 밖) 아래': (850, 2100, 4400, 5400),
       '왼쪽 받침대': (4500, 6000, 4400, 5700), '오른쪽 받침대': (10200, 11600, 4400, 5700), '자 눈금': (3950, 4500, 4300, 4950)}
def load(f): return np.asarray(Image.open(os.path.join(D, f)).convert('L'), dtype=np.float32)
def shift(a, b):
    a = a - a.mean(); b = b - b.mean()
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win); return dx, dy, resp
ref = load(first[3]); out = []
def run(label, fs):
    for g in sorted(fs):
        im = load(fs[g]); parts = []
        for n, (x0, x1, y0, y1) in REG.items():
            dx, dy, rp = shift(ref[y0:y1, x0:x1], im[y0:y1, x0:x1]); parts.append(f'{n} {dx:+6.1f},{dy:+5.1f} ({rp:.2f})')
        s = f'{label} 묶음{g:2d} | ' + ' | '.join(parts); print(s, flush=True); out.append(s)
run('첫장', first)
open(os.path.join(HERE, '_out', 'table_vs_cam.txt'), 'w').write('\n'.join(out))
