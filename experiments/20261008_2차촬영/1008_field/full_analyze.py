# 10-08 본촬영 130장: 시편 분류표 + Ø20 베어링부 폭, 핀 폭, 경계 흐림, 배경 밝기
# 묶음은 사진 시각 간격(25초 초과)으로 나눈다. 순서는 현장 메모(20261008_155336.jpg)와 실측 csv 기준
import sys, os, csv, datetime as dt, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
from t1_analyze import meas, edges  # 같은 경계 규칙(열마다 배경과 어두운 쪽의 중간값)

D = sys.argv[1]
# 묶음 순서 -> (시편, 역할, 등급, 실측mm, 초점구간, 핀배치)
PLAN = [('NO1','cal','하',20.010,'F1','P1'), ('NO2','cal','중',20.020,'F1','P2'), ('NO3','cal','상',20.030,'F1','P2'),
        ('NO4','val','상',20.030,'F1','P2'), ('NO5','val','중',20.020,'F2','P2'), ('NO6','val','하',20.010,'F2','P2'),
        ('NO7','rand','',20.017,'F3','P2'), ('NO8','rand','',20.019,'F3','P2'), ('NO9','rand','',20.015,'F3','P2'),
        ('NO10','rand','',20.015,'F3','P2'), ('NG1','ng','',19.993,'F3','P2'), ('NG2','ng','',20.002,'F3','P2'),
        ('NO1','cal_re','하',20.010,'F3','P2')]
fs = sorted(f for f in os.listdir(D) if f.endswith('.jpg') and f != '20261008_155336.jpg')
ts = [dt.datetime.strptime(f[:15], '%Y%m%d_%H%M%S') for f in fs]
grp, k = [], 0
for i in range(len(fs)):
    if i and (ts[i] - ts[i-1]).seconds > 25: k += 1
    grp.append(k)
assert k + 1 == len(PLAN), (k + 1, len(PLAN))

def land(g, bg):
    # 칼라(폭 600px 넘음)가 끝나는 곳부터 홈(폭이 갑자기 줄어듦) 전까지가 베어링부. 양끝 30px 은 모서리라 뺀다
    xs = np.arange(7600, 8400, 4); ws = []
    for x in xs:
        col = g[3600:4320, x]; t, b = edges(col, (bg + np.percentile(col, 5)) / 2); ws.append(b - t)
    ws = np.array(ws); c = np.where(ws > 600)[0]; xc = xs[c[-1]]
    after = (xs > xc + 30); ref = np.median(ws[after][:20])
    gr = xs[after & (ws < ref - 15)][0]
    return int(xc + 30), int(gr - 30)

out = os.path.join(os.path.dirname(__file__), '_out'); os.makedirs(out, exist_ok=True)
w = csv.writer(open(os.path.join(out, 'full_rows.csv'), 'w', newline='', encoding='utf-8-sig'))
w.writerow(['파일', '시각', 'x0', 'x1', '묶음', '시편', '역할', '등급', '실측mm', '초점', '핀배치',
            '샤프트px', '샤프트sd', '샤프트중심y', '흐림위', '흐림아래', '핀px', '핀sd', '핀중심y', '배경', '포화%'])
for f, t, gi in zip(fs, ts, grp):
    g = np.asarray(Image.open(os.path.join(D, f)).convert('L'), dtype=float)
    b = g[4500:4800, 6800:7600]; bg = b.mean(); sat = (b >= 254).mean() * 100
    x0, x1 = land(g, bg)                                          # 시편마다 축 방향 위치가 달라서 사진마다 찾는다
    sw, ssd, sc, st, sb = meas(g, x0, x1, 3600, 4320, bg)        # Ø20 베어링부(칼라와 홈 사이)
    pw, psd, pc, _, _ = meas(g, 9655, 9700, 3700, 4250, bg)      # 핀 Ø10
    p = PLAN[gi]
    w.writerow([f, t.strftime('%H:%M:%S'), x0, x1, gi + 1, *p, f'{sw:.2f}', f'{ssd:.2f}', f'{sc:.1f}', f'{st:.0f}', f'{sb:.0f}',
                f'{pw:.2f}', f'{psd:.2f}', f'{pc:.1f}', f'{bg:.1f}', f'{sat:.1f}'])
    print(gi + 1, p[0], f, x0, x1, f'{sw:.2f}', f'{pw:.2f}', flush=True)
