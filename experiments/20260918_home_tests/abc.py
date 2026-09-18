# 09-18 집 JPEG A/B/C 테스트: 기본 카메라 2배줌. A=6072 자동 14장, B=3056 프로모드 14장, C=5912 2분 뒤 7장 (D=5배줌 7장은 별도)
import numpy as np, pathlib, json
exec(open("raw_vs_jpeg.py").read().split("xs = np.arange")[0])
def profile(g, xs):   # 재정의: 윗변은 위에서 첫 어두움, 아랫변은 그 아래에서 "밝음이 150행 이상 이어지는 첫 자리"(하이라이트 줄 무시)
    g = cv2.GaussianBlur(g, (5, 5), 0); H = g.shape[0]; out = []
    for x in xs:
        c = g[:, x]; bg = np.median(c[5:int(H*0.06)])
        i = int(H*0.06)
        while i < H-40 and c[i] > bg - 40: i += 1
        j = i + 5; found = False
        while j < H-151:
            if np.all(c[j:j+150] > bg - 40): found = True; break
            j += 1
        if i >= H-40 or not found: out.append(np.nan); continue
        out.append(edge_grad(c, j, -1) - edge_grad(c, i, +1))
    return np.array(out)
D = pathlib.Path("data/20260918_abc_test")
def measure2(g, xs, lo=1450, hi=1800):
    d = profile(g, xs); ok = np.isfinite(d); med = np.nanmedian(d[(xs > lo) & (xs < hi)])
    collar = np.where(ok & (d > med + 40) & (xs < lo))[0]
    if collar.size == 0: return np.nan, None
    cs = collar.max() + 1; j = cs + 10
    while j < len(d) and not (d[j] < med - 15): j += 1
    if j >= len(d): return np.nan, None
    L = j - cs; win = slice(cs + int(L*0.25), j - int(L*0.25))
    return float(np.nanmedian(d[win])), (int(xs[cs]), int(xs[j]))
xs = np.arange(700, 2300, 3)
groups = {"A 자동 6072 2.01x": [], "B 프로 3056 2.04x": [], "C 2분뒤 5912 2.08x": [], "D 5배줌 3056": []}
for f in sorted(D.glob("*.jpg")):
    from PIL import Image; Image.MAX_IMAGE_PIXELS = None
    w = Image.open(f).size[0]; t = int(f.stem[9:11])*3600 + int(f.stem[11:13])*60 + int(f.stem[13:15])
    if w == 6072: k = "A 자동 6072 2.01x"
    elif w == 5912: k = "C 2분뒤 5912 2.08x"
    elif w == 3056 and t < 11*3600+54*60: k = "B 프로 3056 2.04x"
    else: k = "D 5배줌 3056"
    g = norm(load_jpeg(f))
    v, win = measure2(g, np.arange(900, 2700, 3), 1400, 2100) if k.startswith("D") else measure2(g, xs); groups[k].append((t, v, win))
out = {}
for k, rows in groups.items():
    t = np.array([r[0] for r in rows], float); t -= t[0]; v = np.array([r[1] for r in rows]); ok = np.isfinite(v)
    print(k, [round(x,1) if np.isfinite(x) else None for x in v.tolist()])
    if ok.sum() < 3: continue
    t, v = t[ok], v[ok]; um = 20.02e3 / v.mean()
    b, a = np.polyfit(t, v, 1); res = v - (a + b*t)
    print(f"{k:20s} n={len(v):2d} span={t.max():3.0f}s 1px≈{um:.1f}um | 평균 {v.mean():7.2f}px | 원 sd {v.std(ddof=1):.3f}px={v.std(ddof=1)*um:4.1f}um | 추세 {b*60:+.3f}px/분({b*60*um:+.1f}um/분) r={np.corrcoef(t,v)[0,1]:+.2f} | 추세제거 sd {res.std(ddof=1):.3f}px={res.std(ddof=1)*um:4.1f}um | 창 {rows[0][2]}")
    out[k] = dict(t=t.tolist(), v=v.tolist(), um=um)
json.dump(out, open("abc.json", "w"))
