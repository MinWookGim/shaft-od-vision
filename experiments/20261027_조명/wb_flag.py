# EXIF WhiteBalance(0 자동, 1 수동) 사진별 기록과 핀 크기
import glob, os, csv
from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None
T = {v: k for k, v in ExifTags.TAGS.items()}
L = {r['file']: r for r in csv.DictReader(open('light_check.csv'))}
fs = sorted(glob.glob('../../field/photos/20261021_원전기계/3차_261021_분류/[12]_*/NO*/*.jpg'), key=os.path.basename)
w = csv.writer(open('wb_flag.csv', 'w', newline='')); w.writerow(['file', 'stage', 'no', 'wb', 'pin', 'R_B'])
for f in fs:
    n = os.path.basename(f)
    if n not in L: continue
    s = Image.open(f).getexif().get_ifd(0x8769)
    w.writerow([n, L[n]['stage'], L[n]['no'], s.get(T['WhiteBalance']), L[n]['pin'], L[n]['R_B']])
