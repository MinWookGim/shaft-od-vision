# 0917 E4: 프로파일에서 사진별 특징량 -> 시편 평균 -> LOO
# 위치 정렬: 칼라 오른쪽 끝(xs), 홈 왼쪽 끝(xg) 을 자동으로 찾는다 = "측정 위치 라벨"을 완벽히 준 상황의 대용
import json, numpy as np
D = json.load(open("profiles.json")); X = np.array(D["xs"])
TRUE = dict(S01=20.017,S02=20.010,S03=20.030,S04=20.021,S05=20.026,S06=20.012,S07=20.014,
            S08=20.011,S09=20.024,S10=20.017,N01=20.405,N02=20.040,N03=20.006)

def feats(t, b):
    t, b = np.array(t, float), np.array(b, float); d = b - t; c = (b + t) / 2
    ok = np.isfinite(d)
    # 칼라(>620) 구간의 오른쪽 끝, 그 뒤 첫 590 미만 지점
    col = np.where(ok & (d > 620) & (X < 1300))[0]
    xs_i = col.max() + 1
    # 홈: xs 이후 150px 지나서 처음으로 지름이 (측정부 중앙값 - 25) 아래로 떨어지는 곳
    base = np.nanmedian(d[xs_i + 30: xs_i + 90])
    j = xs_i + 75
    while not (d[j] < base - 25): j += 1
    xg_i = j
    seg = lambda a, b_: slice(a, b_)
    L = xg_i - xs_i                                   # 측정부 길이(px 인덱스, 1칸=2px)
    m_al = d[xs_i + int(L * .2): xg_i - int(L * .2)]  # 정렬된 측정 창 (가운데 60%)
    fx = (X >= 1180) & (X < 1300)                     # 고정 픽셀 창
    # 칼라 기준: 칼라 끝에서 왼쪽 20~60칸
    collar = d[xs_i - 60: xs_i - 20]
    # 끝단 기준: 홈 뒤 평평한 곳 (xg 기준 +110~+500칸)
    end_i = np.arange(xg_i + 110, xg_i + 500)
    endv = d[end_i]
    k = np.isfinite(endv)
    slope, icpt = np.polyfit(X[end_i][k], endv[k], 1)
    ang = np.polyfit(X[ok & (X < 2500)], c[ok & (X < 2500)], 1)[0]
    # 측정부 기울기(위치 민감도)
    mk = np.arange(xs_i + int(L * .2), xg_i - int(L * .2))
    mslope = np.polyfit(X[mk], d[mk], 1)[0]
    return dict(A_fix=np.nanmedian(d[fx]), A_al=np.nanmedian(m_al), m_sd=np.nanstd(m_al),
                collar=np.nanmedian(collar), end=np.nanmedian(endv), end_slope=slope,
                ang=ang, mslope=mslope, xs=X[xs_i], xg=X[xg_i],
                x_meas=(X[xs_i] + X[xg_i]) / 2, x_col=X[xs_i - 40], x_end=np.mean(X[end_i]))

rows = []
for f, v in D["prof"].items():
    try:
        r = feats(v["top"], v["bot"]); r.update(spec=v["spec"], file=f); rows.append(r)
    except Exception as e:
        print("실패", v["spec"], f, e)
json.dump(rows, open("feats.json", "w"), default=float)
specs = list(TRUE)
M = {s: {k: np.mean([r[k] for r in rows if r["spec"] == s]) for k in rows[0] if k not in ("spec", "file")} for s in specs}
SD = {s: {k: np.std([r[k] for r in rows if r["spec"] == s], ddof=1) for k in ("A_fix","A_al")} for s in specs}
print("시편  실측    A_fix   A_al  sd_al  collar   end   xs   xg  ang(x1e3) endslope(x1e3) mslope(x1e3)")
for s in specs:
    m = M[s]
    print(f"{s} {TRUE[s]:.3f} {m['A_fix']:7.2f} {m['A_al']:7.2f} {SD[s]['A_al']:5.2f} {m['collar']:7.2f} {m['end']:6.2f} {m['xs']:5.0f} {m['xg']:5.0f} {m['ang']*1e3:6.1f} {m['end_slope']*1e3:6.2f} {m['mslope']*1e3:6.2f}")

def loo(x, sp):
    e = []
    for s in sp:
        tr = [t for t in sp if t != s]
        b, a = np.polyfit([x[t] for t in tr], [TRUE[t] for t in tr], 1)
        e.append((a + b * x[s] - TRUE[s]) * 1000)
    return np.array(e)

# 두 기준 보간: 배율이 x 에 따라 선형으로 변한다고 보고, 측정부 위치에서의 배율을 칼라·끝단 사이 보간
def two_ref(s):
    m = M[s]; w = (m["x_end"] - m["x_meas"]) / (m["x_end"] - m["x_col"])
    return np.log(m["A_al"]) - (w * np.log(m["collar"]) + (1 - w) * np.log(m["end"]))
cands = {
    "A 고정창": {s: M[s]["A_fix"] for s in specs},
    "A 정렬창": {s: M[s]["A_al"] for s in specs},
    "A/끝단": {s: M[s]["A_al"] / M[s]["end"] for s in specs},
    "A/칼라": {s: M[s]["A_al"] / M[s]["collar"] for s in specs},
    "A/(칼라·끝단 보간)": {s: two_ref(s) for s in specs},
    "A x cos(기울기)": {s: M[s]["A_al"] * np.cos(np.arctan(M[s]["ang"])) for s in specs},
}
sets = {"13개": specs, "N01제외12": [s for s in specs if s != "N01"],
        "S04·S07·N01제외10": [s for s in specs if s not in ("N01","S04","S07")]}
T12 = [TRUE[s] for s in sets["N01제외12"]]
print("\n기준선(평균 찍기) N01제외12 LOO MAE = 8.4um")
for name, x in cands.items():
    line = f"{name:18s}"
    for sn, sp in sets.items():
        e = loo(x, sp); r = np.corrcoef([x[s] for s in sp], [TRUE[s] for s in sp])[0, 1]
        line += f" | {sn}: r={r:+.2f} MAE={np.mean(np.abs(e)):5.1f} max={e[np.argmax(np.abs(e))]:+6.1f}"
    print(line)
    # S04/S07 이 정상 무리에서 얼마나 떨어져 있나 (um 환산)
    ref = np.median([x[s] for s in specs if s[0]=="S" and s not in ("S04","S07")])
    scale = 20.02 / ref if name != "A/(칼라·끝단 보간)" else 20.02
    print(f"{'':18s}   정상무리 대비 S04 {(x['S04']-ref)*scale*1000:+7.1f}um  S07 {(x['S07']-ref)*scale*1000:+7.1f}um  N01 {(x['N01']-ref)*scale*1000:+7.1f}um")
