# 0917 E3: 축 방향 지름 프로파일. YOLO/박스 없이 실루엣 윗변·아랫변을 열마다 서브픽셀로 찾는다
# 목적: (1) 측정 위치(박스 x)가 바뀌면 값이 얼마나 변하나 = "위치 라벨"의 가치
#       (2) 기준부가 비스듬히(원근) 찍혔나 = 자기정규화가 S04/S07 을 덜 고친 이유 후보
import csv, json, numpy as np, cv2
SM = "data/sm161/"   # 2992 축소 사진 폴더 (사진은 저장소 밖)
man = list(csv.DictReader(open("experiments/results/20260916/manifest_20260916.csv", encoding="utf-8-sig")))
XS = np.arange(0, 2992, 2)

def cross(prof, start, step, thr):
    # start 에서 step 방향으로 가며 밝음->어둠 첫 교차를 선형보간으로
    i = start
    while 0 < i < len(prof) - 1:
        if prof[i] >= thr and prof[i + step] < thr:
            return i + step * (prof[i] - thr) / (prof[i] - prof[i + step])
        i += step
    return np.nan

def cross_local(prof, start, step, bg):
    # 수정(0917): 윗변은 회색 금속이라 전체 최저 밝기를 쓰면 안쪽을 잡는다 -> 엣지 바로 안쪽 국소 밝기 기준
    i = start
    while 0 < i < len(prof) - 12 and prof[i] >= bg - 20:
        i += step
    if not (0 < i < len(prof) - 12):
        return np.nan
    v_in = np.median(prof[i + 3 * step: i + 10 * step: step])
    return cross(prof, i - 3 * step, step, (bg + v_in) / 2)

def profile(path):
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    top, bot = [], []
    for x in XS:
        c = g[:, x]
        bg1, bg2 = np.median(c[1000:1150]), np.median(c[2100:2200])   # 위·아래 배경 밝기
        t = cross_local(c, 1150, +1, bg1)
        b = cross_local(c, 2150, -1, bg2)
        top.append(t); bot.append(b)
    return np.array(top), np.array(bot)

out = {}
for r in man:
    t, b = profile(SM + r["파일"])
    out[r["파일"]] = dict(spec=r["시편"], top=np.round(t, 3).tolist(), bot=np.round(b, 3).tolist())
    print(r["시편"], r["파일"], flush=True) if r["장번호"] == "1" else None
json.dump(dict(xs=XS.tolist(), prof=out), open("profiles.json", "w"))
