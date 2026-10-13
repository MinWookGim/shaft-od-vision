# NO2 -> NO3 사이에 사진 전체가 움직였나(카메라 이동), 핀·V블록만 움직였나 확인
# 묶음마다 첫 장을 기준 사진(묶음 3 첫 장)과 위상상관으로 비교. 영역별로 따로:
#   왼쪽 V블록·자, 오른쪽 V블록, 가운데 백라이트 아래 테두리, 핀 주변
import os, csv, numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
first = {}
for r in rows: first.setdefault(int(r['묶음']), r['파일'])
REG = {'왼쪽V블록·자': (3900, 6600, 3300, 5900), '오른쪽V블록': (9900, 12400, 3300, 5900),
       '백라이트 아래 테두리': (6600, 9600, 4900, 5600), '핀 주변': (9300, 10100, 3500, 4300)}
def load(f): return np.asarray(Image.open(os.path.join(D, f)).convert('L'), dtype=np.float32)
ref = load(first[3])
out = []
for g in sorted(first):
    im = load(first[g]); line = [f'묶음{g:2d}']
    for name, (x0, x1, y0, y1) in REG.items():
        a, b = ref[y0:y1, x0:x1], im[y0:y1, x0:x1]
        win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        line.append(f'{name} dx {dx:+6.1f} dy {dy:+6.1f}')
    print(' | '.join(line), flush=True); out.append(' | '.join(line))
open(os.path.join(HERE, '_out', 'shift.txt'), 'w').write('\n'.join(out))
