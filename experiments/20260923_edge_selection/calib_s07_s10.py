# 09-23 오전 회의에서 희연님이 요청한 검증:
#   "픽셀당 mm 캘리브레이션 값으로 측정값을 돌려서 시편 7·8·9·10번 실측과 비교해야 정확한 데이터다.
#    오차가 3미크론 이내면 (측정 쪽은) 할 게 없다."
#
# 지금까지 본 건 정밀도(같은 값을 얼마나 일관되게 재는가)뿐이고, 정확도(실제 지름과 얼마나 가까운가)는 안 봤다.
# 이 스크립트는 09-23 수정 설정(vertical_margin=0.05, dia_basis="box")을 원본 해상도에 걸고,
# S01~S10 열 개를 전부 재서 캘리브레이션을 걸고 S07~S10 실측과 대조한다.
#
# ★ 설계에서 피해야 할 함정: 대조할 시편(S07~S10)으로 캘리브까지 잡으면 순환논증이 된다.
#   (자로 잰 값으로 자를 만들고, 그 자로 다시 같은 것을 재는 꼴)
#   그래서 캘리브는 S01~S06 에서만 잡고, S07~S10 은 손대지 않은 채로 맞히게 한다.
from paths import load_core, photo_dir, field_dir, RESULT_OUT
core = load_core()

import csv
import numpy as np
from dataclasses import replace

FIXED = replace(core.Params(), vertical_margin=0.05, dia_basis="box")

# 실측(업체 계측기) — docs/02_facts.md 의 S01~S10
ACTUAL_MM = {
    "S01": 20.017, "S02": 20.010, "S03": 20.030, "S04": 20.021, "S05": 20.026,
    "S06": 20.012, "S07": 20.014, "S08": 20.011, "S09": 20.024, "S10": 20.017,
    # 사이즈 에러품 3개 (7장씩). N01 가공 누락, N02 정삭 누락, N03 은 실측상 정상 범위
    "N01": 20.405, "N02": 20.040, "N03": 20.006,
}
NORMAL = [f"S{i:02d}" for i in range(1, 11)]
NG = ["N01", "N02", "N03"]
TRAIN = ["S01", "S02", "S03", "S04", "S05", "S06"]   # 캘리브를 잡는 데만 쓴다
TEST = ["S07", "S08", "S09", "S10"]                  # 손대지 않고 맞히게 한다


def shots_by_specimen():
    """manifest 에서 시편별 촬영 파일 목록을 읽는다."""
    out = {}
    with open(field_dir() / "manifest_20260916.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["시편"] in ACTUAL_MM:
                out.setdefault(row["시편"], []).append(row["파일"])
    return out


def measure_specimen(files):
    """한 시편의 14장을 각각 재고, 장별 대표값(px)들을 돌려준다."""
    per_shot = []
    for fn in files:
        path = photo_dir() / fn
        box, conf = core.detect_box(path)
        if box is None:
            per_shot.append(np.nan)
            continue
        r = core.measure_roi(core.imread(path), box, FIXED)
        per_shot.append(r["measured_pixel"] if r["ok"] else np.nan)
    return np.array(per_shot, dtype=float)


CACHE = RESULT_OUT / "calib_s07_s10_px.csv"


def load_cache():
    """이미 잰 시편별 px 가 있으면 읽는다 (140장 재측정을 건너뛰기 위함)."""
    if not CACHE.exists():
        return None
    with open(CACHE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    got = {r["시편"]: float(r["측정px"]) for r in rows}
    return got if set(got) == set(ACTUAL_MM) else None


def measure_all():
    """S01~S10 을 14장씩 전부 재고 시편별 대표 px 를 돌려준다 (140장, 몇 분 걸린다)."""
    files = shots_by_specimen()
    px = {}
    print(f"{'시편':>4} | {'장수':>5} | {'대표 px(14장 median)':>20} | {'장간 sd(px)':>11} | 실측 mm")
    print("-" * 74)
    for s in sorted(ACTUAL_MM):
        vals = measure_specimen(files[s])
        ok = vals[~np.isnan(vals)]
        px[s] = float(np.median(ok))
        print(f"{s:>4} | {len(ok):>2}/{len(vals):<2} | {np.median(ok):>20.2f} | {ok.std(ddof=1):>11.3f} | {ACTUAL_MM[s]:.3f}")
    return px


def main(use_cache=True):
    px = load_cache() if use_cache else None
    if px:
        print(f"(이미 측정해 둔 값을 쓴다: {CACHE}. 다시 재려면 --remeasure)\n")
    else:
        px = measure_all()

    # ---- 캘리브레이션 (S01~S06 만 사용) ----
    tr_px = np.array([px[s] for s in TRAIN])
    tr_mm = np.array([ACTUAL_MM[s] for s in TRAIN])

    # 방법 A: 1점 캘리브 — S01 하나로 배율만 잡는다 (현장에서 제일 흔한 방식)
    scale_1pt = ACTUAL_MM["S01"] / px["S01"]          # mm/px

    # 방법 B: 6개 평균으로 배율만 잡는다 (원점을 지나는 직선, 기울기만 적합)
    scale_avg = float(np.sum(tr_px * tr_mm) / np.sum(tr_px ** 2))

    # 방법 C: 2자유도 직선 적합 (mm = a*px + b). 오프셋까지 허용
    a, b = np.polyfit(tr_px, tr_mm, 1)

    print()
    print("=== 캘리브레이션 (S01~S06 로만 잡음, S07~S10 은 안 씀) ===")
    print(f"  A 1점(S01)      : {scale_1pt*1000:.4f} µm/px")
    print(f"  B 6개 배율적합   : {scale_avg*1000:.4f} µm/px")
    print(f"  C 직선적합       : {a*1000:.4f} µm/px, 절편 {b:+.4f} mm")
    print(f"  (참고) 팀 문서 상수: 34.7µm/px @2992 → 원본 환산 {34.7*2992/6008:.2f} µm/px")

    # ---- S07~S10 대조 ----
    for name, pred in [
        ("A 1점(S01)", lambda v: v * scale_1pt),
        ("B 6개 배율", lambda v: v * scale_avg),
        ("C 직선적합", lambda v: a * v + b),
    ]:
        print()
        print(f"=== {name} 캘리브로 S07~S10 맞히기 ===")
        print(f"{'시편':>4} | {'측정 px':>9} | {'예측 mm':>9} | {'실측 mm':>8} | {'오차 µm':>9}")
        errs = []
        for s in TEST:
            mm = pred(px[s])
            err_um = (mm - ACTUAL_MM[s]) * 1000
            errs.append(err_um)
            print(f"{s:>4} | {px[s]:>9.2f} | {mm:>9.4f} | {ACTUAL_MM[s]:>8.3f} | {err_um:>+9.1f}")
        errs = np.array(errs)
        print(f"  MAE {np.abs(errs).mean():.1f}µm | 최대 {np.abs(errs).max():.1f}µm | "
              f"치우침(평균) {errs.mean():+.1f}µm | 희연님 기준 3µm {'통과' if np.abs(errs).max() <= 3 else '미달'}")

    # ---- 시편을 서로 구별할 수 있나 (상관) ----
    names = NORMAL
    all_px = np.array([px[s] for s in names])
    all_mm = np.array([ACTUAL_MM[s] for s in names])
    r = float(np.corrcoef(all_px, all_mm)[0, 1])
    um_per_px = scale_1pt * 1000            # 퇴화한 직선적합(C) 말고 제대로 된 배율을 쓴다
    print()
    print("=== 시편 간 구별력 (정상품 10개) ===")
    print(f"  실측 폭   {(all_mm.max()-all_mm.min())*1000:>6.0f}µm = {(all_mm.max()-all_mm.min())*1000/um_per_px:>6.2f}px 상당")
    print(f"  측정 px 폭 {all_px.max()-all_px.min():>6.2f}px = {(all_px.max()-all_px.min())*um_per_px:>6.0f}µm 상당")
    print(f"  실측 vs 측정 상관 r = {r:+.3f}  (1에 가까울수록 시편 차이를 따라감)")

    # 배치 영향이 기록된 S04·S07 을 빼면 얼마나 달라지나 (docs/02_facts.md)
    keep = [i for i, n in enumerate(names) if n not in ("S04", "S07")]
    r2 = float(np.corrcoef(all_px[keep], all_mm[keep])[0, 1])
    print(f"  S04·S07(배치 영향 기록된 시편) 제외 8개: px 폭 {all_px[keep].max()-all_px[keep].min():.2f}px "
          f"= {(all_px[keep].max()-all_px[keep].min())*um_per_px:.0f}µm, r = {r2:+.3f}")
    print(f"  한 시편을 반복해 재면 장간 sd 0.16~0.65px(3~11µm)인데, 시편을 바꾸면 위만큼 벌어진다")
    print(f"  -> 지금 정확도를 막는 건 엣지 검출이 아니라 시편을 놓는 방법(배치)이다")

    # ---- 규격 판정으로 썼을 때 (Ø20.020 ±0.010 = 20.010~20.030) ----
    LO, HI = 20.010, 20.030
    print()
    print(f"=== 이 측정값을 그대로 합격/불합격 판정에 쓰면 (규격 {LO}~{HI}mm) ===")
    print(f"{'시편':>4} | {'예측 mm':>9} | {'실측 mm':>8} | {'비전 판정':>9} | {'실제':>8} | 일치?")
    hit = miss = false_alarm = 0
    for s_ in NORMAL + NG:
        mm = px[s_] * scale_1pt
        v_ng = not (LO <= mm <= HI)
        t_ng = not (LO <= ACTUAL_MM[s_] <= HI)
        ok = v_ng == t_ng
        if v_ng and t_ng: hit += 1
        elif v_ng and not t_ng: false_alarm += 1
        elif t_ng and not v_ng: miss += 1
        print(f"{s_:>4} | {mm:>9.4f} | {ACTUAL_MM[s_]:>8.3f} | {'불합격' if v_ng else '합격':>9} | "
              f"{'불합격' if t_ng else '합격':>8} | {'O' if ok else 'X'}")
    print(f"  정상품 10개 중 불합격으로 잘못 판정 = {sum(1 for n in NORMAL if not (LO <= px[n]*scale_1pt <= HI))}개")
    print(f"  맞힘 {hit} / 헛경보 {false_alarm} / 놓침 {miss}")

    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    with open(RESULT_OUT / "calib_s07_s10_px.csv", "w", encoding="utf-8") as f:
        f.write("시편,측정px,실측mm\n")
        for s in sorted(ACTUAL_MM):
            f.write(f"{s},{px[s]:.3f},{ACTUAL_MM[s]:.3f}\n")
    print(f"\n시편별 px 저장: {RESULT_OUT / 'calib_s07_s10_px.csv'}")


if __name__ == "__main__":
    import sys
    main(use_cache="--remeasure" not in sys.argv)
