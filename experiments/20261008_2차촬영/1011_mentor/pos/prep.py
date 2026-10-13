# 실험 2 준비: 측정 위치(베어링부·핀) 찾는 작은 YOLO 를 10-08 사진으로 학습하기 위한 데이터셋
# 라벨 = 10-08 분석에서 사진마다 찾은 위치(full_rows.csv)
#   베어링부: x0~x1, 세로 = 샤프트중심y +- 샤프트px/2
#   핀: 어두운 몸통 x 9540~9840, 세로 = 핀중심y +- 핀px/2
# 나누기 = 시편 묶음 단위 2겹(홀수 묶음 / 짝수 묶음). 같은 시편 10장이 train·test 에 섞이지 않게
# 추가 시험 세트
#   shifted: test 사진을 크기 0.7~1.3배, 위치 +-25% 로 옮긴 것(카메라를 옮겼다고 가정, 라벨도 같이 옮김)
#   0916: 다른 날·다른 구도 사진 20장(라벨 없음, 눈으로만 확인)
import os, csv, glob, random, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
P = os.path.join(ROOT, '..', '..', 'field', 'photos')
D = os.path.join(P, '20261008_원전기계', 'full_result')
OUT = os.path.join(ROOT, '_data', 'pos')
rows = list(csv.DictReader(open(os.path.join(ROOT, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
W = 2048                                   # 16320 -> 2048 (약 1/8)
random.seed(0)

def boxes(r):
    cy, h = float(r['샤프트중심y']), float(r['샤프트px'])
    py, ph = float(r['핀중심y']), float(r['핀px'])
    return [(0, int(r['x0']), cy - h / 2, int(r['x1']), cy + h / 2), (1, 9540, py - ph / 2, 9840, py + ph / 2)]

def write(img, bx, split, name):
    os.makedirs(os.path.join(OUT, 'images', split), exist_ok=True); os.makedirs(os.path.join(OUT, 'labels', split), exist_ok=True)
    img.save(os.path.join(OUT, 'images', split, name + '.jpg'), quality=92)
    w, h = img.size
    with open(os.path.join(OUT, 'labels', split, name + '.txt'), 'w') as f:
        for c, a, b, c2, d in bx:
            f.write(f'{c} {(a + c2) / 2 / w:.6f} {(b + d) / 2 / h:.6f} {(c2 - a) / w:.6f} {(d - b) / h:.6f}\n')

meta = []
for r in rows:
    g = int(r['묶음']); fold = 'A' if g % 2 else 'B'
    im = Image.open(os.path.join(D, r['파일'])).convert('RGB')
    s = W / im.size[0]; small = im.resize((W, round(im.size[1] * s)), Image.LANCZOS)
    bx = [(c, a * s, b * s, c2 * s, d * s) for c, a, b, c2, d in boxes(r)]
    name = r['파일'][:-4]
    write(small, bx, fold, name)
    # 위치·크기 옮긴 판 (사진마다 하나)
    k = random.uniform(0.7, 1.3); dx = random.uniform(-.25, .25) * W; dy = random.uniform(-.25, .25) * small.size[1]
    canvas = Image.new('RGB', small.size, (0, 0, 0))
    sm2 = small.resize((round(small.size[0] * k), round(small.size[1] * k)), Image.LANCZOS)
    ox = round((small.size[0] - sm2.size[0]) / 2 + dx); oy = round((small.size[1] - sm2.size[1]) / 2 + dy)
    canvas.paste(sm2, (ox, oy))
    bx2 = [(c, a * k + ox, b * k + oy, c2 * k + ox, d * k + oy) for c, a, b, c2, d in bx]
    if all(0 <= a and c2 <= small.size[0] and 0 <= b and d <= small.size[1] for _, a, b, c2, d in bx2):
        write(canvas, bx2, fold + '_shifted', name)
    meta.append((r['파일'], g, fold, s))
    print(r['파일'], g, fold, flush=True)

# 09-16 다른 구도 20장
os.makedirs(os.path.join(OUT, 'images', 'p0916'), exist_ok=True)
for f in sorted(glob.glob(os.path.join(P, '20260916_원전기계', '*.jpg')))[::10]:
    im = Image.open(f).convert('RGB'); s = W / max(im.size)
    im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS).save(os.path.join(OUT, 'images', 'p0916', os.path.basename(f)), quality=92)

for fold, other in (('A', 'B'), ('B', 'A')):
    # fold A 모델 = A 로 학습, B 로 검증
    with open(os.path.join(OUT, f'train_{fold}.yaml'), 'w') as f:
        f.write(f'path: {OUT}\ntrain: images/{fold}\nval: images/{other}\nnames:\n  0: land\n  1: pin\n')
csv.writer(open(os.path.join(OUT, 'meta.csv'), 'w', newline='')).writerows([('파일', '묶음', 'fold', 'scale')] + meta)
print('done')
