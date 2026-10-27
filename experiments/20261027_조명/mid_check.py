# 중간 밝기 영역을 빛 센서로: V블록 옆면 왼쪽·오른쪽(밝기 약 120), 바닥 켄트지(약 30), 밝은 배경(약 215)
# 박스 안 빛은 백라이트뿐이라 이 면들도 백라이트 빛을 받는다. 밝은 배경보다 폰의 눌러 담기가 덜할 것으로 기대
# 테스트5(1/40·ISO40, 15:02)도 같이 재서, 노출 차이(본시험이 17% 적음)에 영역값이 얼마나 따라가는지(감도) 본다
import glob, os, csv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
R = {'bright': (4290, 4350, 8030, 8330), 'vL': (4200, 4400, 4000, 4200), 'vR': (4200, 4400, 11200, 11400), 'floor': (300, 800, 6000, 10000)}
SH = '../../field/photos/20261021_원전기계/3차_261021_분류/'
fs = sorted(glob.glob('../../ETC/261021/테스트5/*.jpg')) + sorted([f for f in glob.glob(SH + '[12]_*/NO*/*.jpg') if '_1533' not in f], key=os.path.basename)
w = csv.writer(open('mid_check.csv', 'w', newline=''))
w.writerow(['file', 'set'] + [f'{k}_{c}' for k in R for c in 'RGB'])
for f in fs:
    im = np.asarray(Image.open(f).convert('RGB'), dtype=float)
    vals = []
    for k, (y0, y1, x0, x1) in R.items(): vals += list(im[y0:y1, x0:x1].reshape(-1, 3).mean(0))
    st = 'T5' if '테스트5' in f else f.split('/')[-3][0]
    w.writerow([os.path.basename(f), st] + [f'{v:.3f}' for v in vals]); print(os.path.basename(f), flush=True)
