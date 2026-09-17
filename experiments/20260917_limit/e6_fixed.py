# 0917 E6: 검증 반영 — (1) 보간 가중치 w 를 촬영 구성 상수로 고정 (2) 전부 "오차 sd"(기울기1 가정)로 비교
#          (3) 09-16 파이프라인 C 도 같은 잣대로 (4) N02 손 가림 1장 제외 영향
import json, numpy as np
TRUE = dict(S01=20.017,S02=20.010,S03=20.030,S04=20.021,S05=20.026,S06=20.012,S07=20.014,
            S08=20.011,S09=20.024,S10=20.017,N01=20.405,N02=20.040,N03=20.006)
rows_all = json.load(open("feats.json"))
V = json.load(open("variants_20260916.json"))
specs = list(TRUE)
G12 = [s for s in specs if s != "N01"]
G10 = [s for s in G12 if s not in ("S04", "S07")]

def run(rows, tag):
    M = {s: {k: np.mean([r[k] for r in rows if r["spec"] == s]) for k in rows[0] if k not in ("spec","file")} for s in specs}
    W = np.mean([(M[s]["x_end"]-M[s]["x_meas"])/(M[s]["x_end"]-M[s]["x_col"]) for s in specs])
    two = lambda s, w: np.exp(np.log(M[s]["A_al"]) - (w*np.log(M[s]["collar"]) + (1-w)*np.log(M[s]["end"])))
    C09 = {s: np.mean([r["C"] for r in V[s]["rows"] if r.get("C")]) for s in specs}
    cands = {"A 원시(프로파일)": lambda s: M[s]["A_al"],
             "A/끝단(프로파일)": lambda s: M[s]["A_al"]/M[s]["end"],
             "C 09-16 파이프라인": lambda s: C09[s],
             "A/칼라": lambda s: M[s]["A_al"]/M[s]["collar"],
             "두기준 w시편별": lambda s: two(s, (M[s]["x_end"]-M[s]["x_meas"])/(M[s]["x_end"]-M[s]["x_col"])),
             "두기준 w고정": lambda s: two(s, W)}
    print(f"\n==== {tag}  (w 고정값 {W:.4f})")
    for gname, g in (("정상10+미세NG2 (12)", G12), ("S04·S07 뺀 10", G10)):
        t = np.array([TRUE[s] for s in g]) * 1000
        base = np.mean([abs(TRUE[s]-np.mean([TRUE[u] for u in g if u != s]))*1000 for s in g])
        print(f"[{gname}] 실측 sd {t.std(ddof=1):.1f}  평균찍기 LOO {base:.1f}")
        for n, f in cands.items():
            v = np.array([f(s) for s in g]); ref = np.array([f(s) for s in G10]).mean()
            um = v / ref * 20020
            err = (um - um.mean()) - (t - t.mean())
            e = []
            for i, s in enumerate(g):
                k = [j for j in range(len(g)) if j != i]
                b, a = np.polyfit(v[k], t[k], 1); e.append(a + b*v[i] - t[i])
            x = {s: (f(s)/ref*20020 - 20020) for s in ("S04","S07","N01")}
            extra = f"  S04 {x['S04']:+6.1f} S07 {x['S07']:+7.1f} N01 {x['N01']:+6.1f}" if g is G12 else ""
            print(f"  {n:18s} 오차sd {err.std(ddof=1):6.1f}  LOO MAE {np.mean(np.abs(e)):5.1f}  r {np.corrcoef(v,t)[0,1]:+.2f}{extra}")

run(rows_all, "전체 161장")
run([r for r in rows_all if r["file"] != "20260916_155806.jpg"], "N02 손 가림 1장 제외")
# 중심선 높이(측정부 위치) — 검증자 수치 대조용
