# 09-23 추가: 제안서(09-17) 숫자와 같은 잣대로 이번 측정값을 환산한다.
#
# 제안서 17µm 는 "오차 sd" 다 (experiments/20260917_limit 의 e6_fixed.py 와 같은 식):
#   측정값을 S04·S07 뺀 정상 10개 평균 = 20020µm 로 맞춰 µm 로 바꾸고,
#   (측정 - 측정평균) - (실측 - 실측평균) 의 표준편차. 즉 공통으로 밀린 양(바이어스)은 뺀 흔들림이다.
# 이번 측정(원본 6008px + 수정 설정)은 측정부만 재므로, 비교할 수 있는 건 "측정부 그대로" 행뿐이다.
# (17µm 는 칼라·끝단 두 기준 보간 값이라 칼라·끝단을 따로 재야 한다)
#
# 측정은 다시 하지 않는다. calib_s07_s10.py 가 저장한 시편별 px 를 쓴다.
from paths import RESULT_OUT

import csv
import numpy as np

CACHE = RESULT_OUT / "calib_s07_s10_px.csv"
SPECS = [f"S{i:02d}" for i in range(1, 11)] + ["N01", "N02", "N03"]
G12 = [s for s in SPECS if s != "N01"]                 # 정상 10 + 미세 NG 2 (제안서의 "12개")
G10 = [s for s in G12 if s not in ("S04", "S07")]      # 배치 영향 기록된 둘을 뺀 10개

# 09-17 e6_fixed.py 결과 (측정부 그대로, 2992px 프로파일) — 대조용
REF_0917 = {"12": 189, "10": 32, "S04": -329, "S07": 523, "N01": 370}


def main():
    with open(CACHE, encoding="utf-8") as f:
        data = {r["시편"]: (float(r["측정px"]), float(r["실측mm"])) for r in csv.DictReader(f)}
    ref = np.mean([data[s][0] for s in G10])            # 이 평균을 20020µm 로 본다

    print("=== proposal_metric.py — 제안서 잣대(오차 sd)로 환산, 측정부 그대로 (09-23 추가) ===")
    for name, group in (("12", G12), ("10", G10)):
        px = np.array([data[s][0] for s in group])
        true_um = np.array([data[s][1] for s in group]) * 1000
        meas_um = px / ref * 20020
        err = (meas_um - meas_um.mean()) - (true_um - true_um.mean())   # 공통 바이어스 제거
        r = np.corrcoef(px, true_um)[0, 1]
        print(f"  {name}개 오차 sd {err.std(ddof=1):6.1f}µm  (09-17: {REF_0917[name]}µm)  r {r:+.2f}")
    for s in ("S04", "S07", "N01"):
        print(f"  {s} 차이 {data[s][0] / ref * 20020 - 20020:+7.1f}µm  (09-17: {REF_0917[s]:+d}µm)")


if __name__ == "__main__":
    main()
