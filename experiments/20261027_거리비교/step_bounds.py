# 선행 개발자 ROI [8008, 3329, 8280, 4276]가 Ø20 단 전체(10.2mm)인지 확인: 본시험 NO2 첫 장, 열별 지름
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
exec(open('dist_compare.py').read().split("rows = {}")[0])
f = '../../field/photos/20261021_원전기계/3차_261021_분류/1_본시험_암막/NO02/20261021_153554.jpg'
g = np.asarray(Image.open(f).convert('L'), float)
for x in range(7900, 8460, 20):
    d = height(g, x, 3150, 4350)[0]
    print(x, f'{d:.1f}', '<- ROI' if 8008 <= x <= 8280 else '')
