import numpy as np
rng = np.random.default_rng(0)

mad  = np.array([1157.00,1157.00,1157.07,1157.01,1157.00,1157.64,1157.17,1157.42,1157.00,1157.31,1157.62,1157.22,1158.00,1157.62])
line = np.array([1156.38,1156.44,1157.21,1157.29,1156.83,1157.19,1157.30,1156.89,1156.48,1157.30,1156.98,1157.37,1157.95,1157.15])
n = len(mad)
print(f"n={n}  sd(MAD)={mad.std(ddof=1):.3f}px   sd(라인피팅)={line.std(ddof=1):.3f}px")

# 같은 14장을 짝으로 부트스트랩 (장 단위 재표집)
diffs = []
for _ in range(20000):
    idx = rng.integers(0, n, n)
    diffs.append(line[idx].std(ddof=1) - mad[idx].std(ddof=1))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"sd 차이(라인-MAD) = {line.std(ddof=1)-mad.std(ddof=1):+.3f}px")
print(f"  부트스트랩 95% 구간: {lo:+.3f} ~ {hi:+.3f}px  -> 0을 {'포함(구분 불가)' if lo<0<hi else '미포함(유의)'}")
print(f"  라인피팅이 더 나쁠 확률 = {(diffs>0).mean()*100:.0f}%")

# sd 자체의 불확실성(대략): sd/sqrt(2(n-1))
for name, v in [("MAD", mad), ("라인피팅", line)]:
    s = v.std(ddof=1); se = s/np.sqrt(2*(n-1))
    print(f"  {name:6s} sd={s:.3f}px ± {se:.3f}px  (약 {s*17.28:.1f} ± {se*17.28:.1f} um)")

# 기본설정 A2에서 '정답 14개'만의 산포
ok = np.array([1156.4,1155.2,1156.9,1158.3,1154.1,1154.1,1157.1,1157.5,1154.0,1156.9,1156.1,1153.9,1154.8,1154.5])
print(f"\n참고: 기본설정 A2에서 정답 14개만의 지름 sd = {ok.std(ddof=1):.3f}px (median {np.median(ok):.1f}) "
      f"vs 수정설정 26점 std 0.932px")
