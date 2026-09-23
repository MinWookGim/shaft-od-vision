# 09-23 추가 확인: 팀 아이디어 "YOLO 박스 안에서만 엣지를 찾자"(= vertical_margin 0)가 원래 맞았나?
#
# test_box_restrict.py 의 ⑤(margin 0)는 866px 오답이었다. 그런데 그때는 허용 지름 범위가
# ROI 높이 기준(dia_basis="roi")이라, 영역을 박스로 좁히자 범위도 362~868px로 같이 줄어 진짜(1156px)가 잘렸다.
# 그래서 허용 범위만 박스 기준으로 떼어낸 ⑥(margin 0 + dia_basis="box")을 채택안 ④(margin 0.05 + box)와 나란히 잰다.
from paths import load_core, photo_dir, field_dir
core = load_core()

import csv
import numpy as np
from dataclasses import replace

CONFIGS = [
    ("④ margin 0.05 + box (채택안)", replace(core.Params(), vertical_margin=0.05, dia_basis="box")),
    ("⑥ margin 0    + box (박스 그대로)", replace(core.Params(), vertical_margin=0.0, dia_basis="box")),
]


def s01_files():
    """manifest 에서 시편1 14장 파일명을 순서대로 읽는다 (2번째가 A2)."""
    with open(field_dir() / "manifest_20260916.csv", encoding="utf-8-sig") as f:
        return [r["파일"] for r in csv.DictReader(f) if r["시편"] == "S01"]


def main():
    files = s01_files()
    print("=== margin0_box.py — 박스 그대로(margin 0)에 허용 범위만 박스 기준으로 (09-23 추가) ===")
    for label, p in CONFIGS:
        meds, stds, valid = [], [], []
        for fn in files:
            path = photo_dir() / fn
            box, conf = core.detect_box(path)
            r = core.measure_roi(core.imread(path), box, p)
            meds.append(r["measured_pixel"] if r["ok"] else np.nan)
            stds.append(r.get("std_pixel", np.nan))
            valid.append(r.get("n_valid", 0))
        m = np.array(meds)
        # A2 = 14장 중 2번째 장. 장간 sd 는 test_all14.py 와 같게 np.std(분모 n)
        print(f"{label} | A2 {m[1]:.2f}px 산포 {stds[1]:.3f}px 유효 {valid[1]}/31 | "
              f"14장 범위 {np.nanmin(m):.1f}~{np.nanmax(m):.1f}px 장간 sd {np.nanstd(m):.3f}px 실패 {int(np.isnan(m).sum())}")


if __name__ == "__main__":
    main()
