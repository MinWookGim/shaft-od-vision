# 실험 1 준비: 사진마다 베어링부 가운데를 중심으로 원본 해상도 1280x1280 을 잘라 저장
# (학습 때 이 안에서 1024 를 무작위로 잘라 256 으로 줄인다 = 위치 흔들기 보강)
import os, csv, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
D = os.path.join(ROOT, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(ROOT, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
S = 1280
crops, info = [], []
for r in rows:
    g = np.asarray(Image.open(os.path.join(D, r['파일'])).convert('L'))
    cx = (int(r['x0']) + int(r['x1'])) // 2; cy = int(round(float(r['샤프트중심y'])))
    crops.append(g[cy - S // 2: cy + S // 2, cx - S // 2: cx + S // 2].copy())
    info.append((r['파일'], int(r['묶음']), r['시편'], r['역할'], r['초점'], float(r['샤프트px']), float(r['배경'])))
    print(r['파일'], flush=True)
os.makedirs(os.path.join(ROOT, '_data'), exist_ok=True)
np.savez_compressed(os.path.join(ROOT, '_data', 'ad_crops.npz'), crops=np.stack(crops), info=np.array(info, dtype=object))
print('saved')
