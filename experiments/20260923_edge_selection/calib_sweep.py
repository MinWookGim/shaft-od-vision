# 09-23 추가 질문: "캘리브를 시편 1개(S01)로만 잡아서 정확도가 나쁜 것 아닌가? 캘리브 시편을 늘리면?"
#
# 캘리브(µm/px)는 숫자 하나다. 시편을 늘리면 그 숫자 하나를 더 정확히 추정할 뿐이다.
# 그래서 두 가지를 본다.
#   1) 캘리브 시편 수 k 를 1~9개로 늘려가며, 모든 조합에 대해 "나머지 시편"을 맞혀본다 (순환논증 없음)
#   2) 오라클: 정상품 전부로 배율을 잡고 같은 시편을 맞힌다. 캘리브를 아무리 늘려도 이보다 좋아질 수 없는 상한
# 오라클도 3µm에 못 미치면, 병목은 캘리브 개수가 아니라 시편마다 px 가 흔들리는 것(배치)이다.
#
# 측정은 다시 하지 않는다. calib_s07_s10.py 가 저장한 시편별 대표 px(results/20260923/calib_s07_s10_px.csv)를 쓴다.
from paths import RESULT_OUT

import csv
import itertools
import numpy as np

CACHE = RESULT_OUT / "calib_s07_s10_px.csv"
NORMAL = [f"S{i:02d}" for i in range(1, 11)]


def load():
    """시편별 (대표 px, 실측 mm) 를 읽는다."""
    with open(CACHE, encoding="utf-8") as f:
        return {r["시편"]: (float(r["측정px"]), float(r["실측mm"])) for r in csv.DictReader(f)}


def fit_scale(data, names):
    """원점을 지나는 직선(배율만)으로 mm/px 를 잡는다. calib_s07_s10.py 방법 B 와 같은 식."""
    px = np.array([data[n][0] for n in names])
    mm = np.array([data[n][1] for n in names])
    return float(np.sum(px * mm) / np.sum(px ** 2))


def errors_um(data, names, scale):
    """배율로 예측한 지름과 실측의 차이(µm)."""
    return np.array([(data[n][0] * scale - data[n][1]) * 1000 for n in names])


def sweep(data, names, label):
    print(f"\n--- {label} ({len(names)}개) ---")
    print(f"{'k':>2} | {'조합':>4} | {'예측 MAE 평균':>12} | {'MAE 최저~최고':>15} | 3µm 이내 비율")
    for k in range(1, len(names)):
        maes, within = [], []
        for train in itertools.combinations(names, k):
            test = [n for n in names if n not in train]   # 캘리브에 안 쓴 시편만 맞힌다
            e = errors_um(data, test, fit_scale(data, train))
            maes.append(np.abs(e).mean())
            within.append(np.mean(np.abs(e) <= 3))
        print(f"{k:>2} | {len(maes):>4} | {np.mean(maes):>10.1f}µm | "
              f"{min(maes):>6.1f}~{max(maes):>6.1f}µm | {np.mean(within) * 100:5.1f}%")

    # 오라클: 전부로 잡고 전부를 맞힌다 (캘리브 개수를 늘려서 얻을 수 있는 최선)
    e = errors_um(data, names, fit_scale(data, names))
    print(f"오라클(전부로 배율 잡고 같은 시편 맞힘): MAE {np.abs(e).mean():.1f}µm, 최대 {np.abs(e).max():.1f}µm, "
          f"3µm 이내 {int(np.sum(np.abs(e) <= 3))}/{len(e)}")
    print("  시편별 오차 µm:", " ".join(f"{n} {x:+.0f}" for n, x in zip(names, e)))


def main():
    data = load()
    print("=== calib_sweep.py — 캘리브 시편 수를 늘리면 정확도가 좋아지나 (09-23 추가) ===")
    print("배율만 잡는 캘리브, 캘리브에 쓴 시편은 맞히는 대상에서 뺌")
    sweep(data, NORMAL, "정상품 전체")
    sweep(data, [n for n in NORMAL if n not in ("S04", "S07")], "S04·S07(배치 영향 기록) 제외")


if __name__ == "__main__":
    main()
