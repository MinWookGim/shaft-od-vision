# 10-27 팀장 가설: 백라이트가 불안정해서(밝기·색·깜빡임) 촬영 중 크기가 변했나? 기존 3차 사진(본시험+재장착 150장)으로 확인
# 사진마다: (1) 배경 밝기 R·G·B (샤프트 아래쪽 배경, 핀과 샤프트 사이 틈) (2) 색 비율 R/B (3) 틈 세로줄의 줄무늬(LED 깜빡임이면 가로 줄무늬)
# 그리고 같은 사진의 핀 px(1021_main/local_edges.csv)와 시간에 따라 같이 움직이는지 본다
import csv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SHARE = '../../field/photos/20261021_원전기계/3차_261021_분류/'
ed = {r['file']: r for r in csv.DictReader(open('../1021_main/local_edges.csv'))}
out = csv.writer(open('light_check.csv', 'w', newline=''))
out.writerow(['file', 'stage', 'no', 'time', 'bgR', 'bgG', 'bgB', 'gapG', 'R_B', 'band_sd', 'band_peak_period', 'pin'])
import glob, os
fs = sorted(glob.glob(SHARE + '1_*/NO*/*.jpg') + glob.glob(SHARE + '2_*/NO*/*.jpg'), key=os.path.basename)
for f in fs:
    n = os.path.basename(f)
    if '_1533' in n: continue                     # NO1은 줌·노출이 달라 뺀다
    im = np.asarray(Image.open(f).convert('RGB'), dtype=float)
    bg = im[4290:4350, 8030:8330].reshape(-1, 3).mean(0)   # Ø20 단 바로 옆 배경(원본 기준 아래 = 실제 위)
    gap = im[3200:4280, 6020:6100, 1]                       # 핀과 샤프트 사이 틈(초록 채널), 세로로 긴 배경
    rows = gap.mean(1)
    trend = np.convolve(rows, np.ones(101) / 101, mode='same')
    res = (rows - trend)[60:-60]
    spec = np.abs(np.fft.rfft(res - res.mean())); spec[:3] = 0
    per = len(res) / np.argmax(spec) if spec.max() > 0 else np.nan
    t = n[9:15]
    out.writerow([n, f.split('/')[-3][0], f.split('/')[-2], t] + [f'{v:.2f}' for v in bg] +
                 [f'{gap.mean():.2f}', f'{bg[0] / bg[2]:.4f}', f'{res.std():.3f}', f'{per:.1f}', ed[n]['pin']])
    print(n, flush=True)
