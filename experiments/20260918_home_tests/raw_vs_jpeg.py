# 09-18 집 RAW 테스트: 같은 프레임의 JPEG vs RAW(DNG) 반복 흔들림 비교. 수동 노출 고정 15쌍, 모니터 백라이트, NG 시편 1개
# 방법: YOLO 없이 열마다 위·아래 엣지를 기울기 최대(±8px 창, 3점 포물선)로 잡아 지름 프로파일 → 칼라끝·홈으로 정렬한 측정 창의 중앙값 → 프레임 간 sd
import glob, numpy as np, cv2, tifffile, json, pathlib
P = pathlib.Path("data/20260918_raw_test")
W = 2992
def to_gray_resized(arr_gray_float, w=W):
    h = int(round(arr_gray_float.shape[0] * w / arr_gray_float.shape[1]))
    return cv2.resize(arr_gray_float, (w, h), interpolation=cv2.INTER_AREA)
def load_jpeg(f): return to_gray_resized(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY).astype(np.float32))
def load_raw(f, half=True):
    # 삼성 200MP DNG 1.7 = JPEG XL 압축 리니어 DNG(이미 디모자이크된 14비트 RGB, WhiteLevel 16383). LibRaw 불가 → tifffile+imagecodecs
    rgb = tifffile.TiffFile(str(f)).pages[0].asarray().astype(np.float32)
    g = rgb.mean(axis=2)
    if half: g = cv2.resize(g, (g.shape[1]//2, g.shape[0]//2), interpolation=cv2.INTER_AREA)
    return g
def norm(g):  # 배경(위쪽 모니터) 중앙값을 225로 맞춤 — 엣지 탐색 문턱을 8비트 감각으로 쓰기 위해
    bg = np.median(g[int(g.shape[0]*0.05):int(g.shape[0]*0.12), :]); return g * (225.0 / bg)
def edge_grad(c, i0, step):
    a, b = sorted((i0-8*step, i0+8*step)); a = max(a, 1); b = min(b, len(c)-2)
    gr = np.abs(np.diff(c[a:b+1])); k = int(np.argmax(gr))
    if 0 < k < len(gr)-1:
        d = gr[k-1] - 2*gr[k] + gr[k+1]; off = 0.5*(gr[k-1]-gr[k+1])/d if d else 0.0
    else: off = 0.0
    return a + k + 0.5 + off
def profile(g, xs):
    g = cv2.GaussianBlur(g, (5, 5), 0); H = g.shape[0]; out = []
    for x in xs:
        c = g[:, x]; bg = np.median(c[int(H*0.05):int(H*0.12)])
        i = int(H*0.12)
        while i < H-20 and c[i] > bg - 40: i += 1
        j = int(H*0.62)
        while j > 20 and c[j] > bg - 40: j -= 1
        if i >= H-20 or j <= 20 or j <= i: out.append(np.nan); continue
        out.append(edge_grad(c, j, -1) - edge_grad(c, i, +1))
    return np.array(out)
def measure(g, xs, scale=1.0):
    d = profile(g, (xs*scale).astype(int)) / scale   # scale: 이미지가 2992 대비 몇 배 큰지
    # 칼라(굵음) 오른쪽 끝과 홈(가늘어짐)을 자동으로 찾는다
    ok = np.isfinite(d); med = np.nanmedian(d[(xs > 1560) & (xs < 1700)])   # 이 구도의 측정부 ≈ x 1550~1740
    collar = np.where(ok & (d > med + 40) & (xs < 1700))[0]
    if collar.size == 0: return np.nan, med, None
    cs = collar.max() + 1
    j = cs + 10
    while j < len(d) and not (d[j] < med - 12): j += 1
    if j >= len(d): return np.nan, med, None
    L = j - cs; win = slice(cs + int(L*0.25), j - int(L*0.25))
    return float(np.nanmedian(d[win])), med, (int(xs[cs]), int(xs[j]))
xs = np.arange(1150, 2300, 3)
rows = []
for jpg in sorted(P.glob("*.jpg")):
    dng = jpg.with_suffix(".dng")
    gj = norm(load_jpeg(jpg)); vj, _, winj = measure(gj, xs)
    graw = load_raw(dng, half=True)                    # 8160 x 6120 (half)
    gr2992 = norm(to_gray_resized(graw)); vr, _, winr = measure(gr2992, xs)
    grh = norm(graw); sc = graw.shape[1] / W; vrh, _, _ = measure(grh, xs, scale=sc)   # RAW half 해상도에서 직접 (px는 2992 기준으로 환산)
    rows.append(dict(f=jpg.name, jpeg=vj, raw2992=vr, raw_half=vrh, win=winj))
    print(f"{jpg.name}  JPEG {vj:8.2f}  RAW→2992 {vr:8.2f}  RAW half {vrh:8.2f}  창 {winj}", flush=True)
json.dump(rows, open("raw_vs_jpeg.json", "w"))
A = {k: np.array([r[k] for r in rows if np.isfinite(r[k])]) for k in ("jpeg", "raw2992", "raw_half")}
um = 20.02e3 / np.nanmean(A["jpeg"])   # 이 시편 지름을 약 20.02mm로 두고 환산(절대값 아님, 배율만)
print(f"\n1px ≈ {um:.1f}um (2992 기준)   n = {len(A['jpeg'])}")
for k, v in A.items():
    print(f"{k:9s} 평균 {v.mean():8.2f}px  sd {v.std(ddof=1):.3f}px = {v.std(ddof=1)*um:5.1f}um   범위 {v.max()-v.min():.2f}px")
d = A["raw2992"] - A["jpeg"]; print(f"RAW−JPEG 프레임별 차이: 평균 {d.mean():+.2f}px  sd {d.std(ddof=1):.3f}px")
