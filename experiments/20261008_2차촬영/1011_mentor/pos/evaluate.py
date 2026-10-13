# 실험 2 평가 (규칙은 돌리기 전에 고정)
#   1) 같은 구도 시험(학습에 안 쓴 시편): 찾았나, 상자 가장자리 오차(원본 px), 그리고
#      그 상자로 지름을 쟀을 때 라벨 상자로 잰 값과 몇 um 다른가 (지름 = 샤프트px / 핀px x 10mm)
#   2) 옮긴 사진(크기 0.7~1.3배, 위치 +-25%): 찾았나, 상자 오차
#   3) 09-16 다른 날·다른 구도: 라벨 없음 -> 그림으로 확인
#   비교: 기존 YOLO(demo_yolo11n_20260917.pt), CPU 한 장 추론 시간
import os, sys, csv, time, glob, json, numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '..', '1008_field')); from t1_analyze import meas
DS = os.path.join(ROOT, '_data', 'pos'); OUT = os.path.join(ROOT, '_out')
FULL = os.path.join(ROOT, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = {r['파일'][:-4]: r for r in csv.DictReader(open(os.path.join(ROOT, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig'))}
S = 16320 / 2048

def labels(split, name, w, h):
    b = {}
    for l in open(os.path.join(DS, 'labels', split, name + '.txt')):
        c, x, y, bw, bh = map(float, l.split())
        b[int(c)] = np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return b

def predict(m, path):
    r = m.predict(path, imgsz=1280, conf=0.25, verbose=False, device=0)[0]
    best = {}
    for c, p, xy in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        if int(c) not in best or p > best[int(c)][0]: best[int(c)] = (p, xy)
    return {c: v[1] for c, v in best.items()}

def diam(g, land, pin, bg):
    """원본 해상도 상자 -> 지름 mm. 베어링부 상자 안 열들, 세로는 상자 중심 +-360 (10-08 분석과 같은 창)"""
    x0, x1 = int(land[0]), int(land[2]); cy = (land[1] + land[3]) / 2
    sw = meas(g, x0, x1, int(cy - 360), int(cy + 360), bg)[0]
    pc = (pin[0] + pin[2]) / 2; py = (pin[1] + pin[3]) / 2
    pw = meas(g, int(pc - 22), int(pc + 23), int(py - 275), int(py + 275), bg)[0]
    return sw / pw * 10

res = {}
for fold, test in (('A', 'B'), ('B', 'A')):
    m = YOLO(os.path.join(ROOT, 'runs', f'pos_{fold}', 'weights', 'best.pt'))
    for split in (test, test + '_shifted'):
        recs = []
        for f in sorted(glob.glob(os.path.join(DS, 'images', split, '*.jpg'))):
            name = os.path.basename(f)[:-4]; w, h = Image.open(f).size
            lab, pr = labels(split, name, w, h), predict(m, f)
            rec = dict(name=name, found=[c in pr for c in (0, 1)])
            rec['err_px'] = [float(np.abs(pr[c] - lab[c]).max() * S) if c in pr else None for c in (0, 1)]
            if not split.endswith('shifted') and all(rec['found']):
                r = rows[name]; g = np.asarray(Image.open(os.path.join(FULL, name + '.jpg')).convert('L'), dtype=float)
                bg = g[4500:4800, 6800:7600].mean()
                d_pred = diam(g, pr[0] * S, pr[1] * S, bg); d_lab = diam(g, lab[0] * S, lab[1] * S, bg)
                rec['d_um'] = (d_pred - d_lab) * 1000
            recs.append(rec); print(fold, split, rec, flush=True)
        res[f'{fold}->{split}'] = recs

# 요약
summ = []
for k, recs in res.items():
    n = len(recs); fl = sum(r['found'][0] for r in recs); fp = sum(r['found'][1] for r in recs)
    el = [r['err_px'][0] for r in recs if r['err_px'][0] is not None]; ep = [r['err_px'][1] for r in recs if r['err_px'][1] is not None]
    du = [abs(r['d_um']) for r in recs if 'd_um' in r]
    summ.append(dict(시험=k, 장수=n, 베어링부찾음=fl, 핀찾음=fp,
                     베어링부_가장자리오차_중앙px=round(float(np.median(el)), 1) if el else None, 최대px=round(float(np.max(el)), 1) if el else None,
                     핀_가장자리오차_중앙px=round(float(np.median(ep)), 1) if ep else None,
                     지름차이_중앙um=round(float(np.median(du)), 2) if du else None, 지름차이_최대um=round(float(np.max(du)), 2) if du else None))
    print(summ[-1])

# 기존 YOLO 와 CPU 시간
old = YOLO(os.path.join(ROOT, '..', '..', 'field', 'models', 'demo_yolo11n_20260917.pt'))
imgs = sorted(glob.glob(os.path.join(DS, 'images', 'B', '*.jpg')))
oldfound = sum(len(old.predict(f, imgsz=640, conf=0.25, verbose=False, device=0)[0].boxes) > 0 for f in imgs)
m = YOLO(os.path.join(ROOT, 'runs', 'pos_A', 'weights', 'best.pt'))
m.predict(imgs[0], imgsz=1280, device='cpu', verbose=False); t = time.time()
for f in imgs[:10]: m.predict(f, imgsz=1280, device='cpu', verbose=False)
cpu = (time.time() - t) / 10
summ.append(dict(기존YOLO_B세트_무언가찾은장수=f'{oldfound}/{len(imgs)}', 새모델_CPU_한장초=round(cpu, 3)))
print(summ[-1])

# 09-16 그림
fig = []
for f in sorted(glob.glob(os.path.join(DS, 'images', 'p0916', '*.jpg')))[:6]:
    im = Image.open(f).convert('RGB'); d = ImageDraw.Draw(im)
    r = m.predict(f, imgsz=1280, conf=0.1, verbose=False, device=0)[0]
    for c, p, xy in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        d.rectangle(list(xy), outline=(255, 0, 0) if c == 0 else (0, 255, 0), width=4); d.text((xy[0], xy[1] - 12), f'{int(c)} {p:.2f}', fill=(255, 0, 0))
    fig.append(im.resize((500, 500)))
canvas = Image.new('RGB', (1500, 1000), 'white')
for i, im in enumerate(fig): canvas.paste(im, ((i % 3) * 500, (i // 3) * 500))
canvas.save(os.path.join(OUT, 'pos_0916.png'))
summ.append(dict(p0916_conf0_1_검출=[len(m.predict(f, imgsz=1280, conf=0.1, verbose=False, device=0)[0].boxes)
                                      for f in sorted(glob.glob(os.path.join(DS, 'images', 'p0916', '*.jpg')))]))
json.dump(dict(summary=summ, detail=res), open(os.path.join(OUT, 'pos_result.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print('done')
