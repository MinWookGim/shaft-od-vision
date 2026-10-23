# 3차 현장 사진 EXIF + 썸네일 밝기 스캔 -> 시편 분류용 표
import sys, glob, os, csv
from PIL import Image, ExifTags
import numpy as np
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, '2026*_??????.jpg'))):
    im = Image.open(p); ex = im.getexif(); sub = ex.get_ifd(0x8769)
    t = sub.get(36867); ss = sub.get(37521, ''); et = sub.get(33434); iso = sub.get(34855)
    ev = sub.get(37380); ori = ex.get(274)
    im.draft('L', (im.width // 8, im.height // 8)); g = np.asarray(im.convert('L'), dtype=float)
    rows.append([os.path.basename(p), t, ss, f'{im.size[0]}x{im.size[1]}',
                 f'1/{round(1/float(et))}' if et else '', iso, float(ev) if ev is not None else '', ori,
                 os.path.getsize(p) // 1000, round(g.mean(), 1), round(np.percentile(g, 99), 0)])
w = csv.writer(sys.stdout, delimiter='\t')
w.writerow(['file', 'dt', 'sub', 'size', 'shutter', 'iso', 'ev', 'ori', 'kB', 'mean', 'p99'])
w.writerows(rows)
