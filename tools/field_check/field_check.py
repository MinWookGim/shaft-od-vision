# -*- coding: utf-8 -*-
"""
10-02 현장 즉석 확인 스크립트

사진 1장(또는 여러 장)을 넣으면 아래를 뽑는다.
  - 측정부 세로 폭 px (샤프트 지름)
  - 핀 폭 px (세운 핀 = 가로 폭, 눕힌 핀 = 세로 폭)
  - 배경 포화 비율 (백라이트가 하얗게 날아간 비율)
그리고 체크리스트 A4(연속 사진의 핀 px 흔들림)와 A5(핀으로 환산한 지름 vs 실측)를 판정한다.

YOLO 를 안 쓴다. 현장에서는 폰이 고정이라 측정부·핀의 위치가 사진마다 같으므로,
A3 때 한 번 좌표(ROI)를 정해 설정 파일에 적어 두고 그 뒤로는 그 칸만 본다.

사용 순서
  1) 좌표 잡기:  python field_check.py grid 사진.jpg
       -> 사진_grid.jpg 가 생긴다(원본 px 좌표 눈금). 이걸 보고 site.json 에 ROI 를 적는다
  2) 확인:       python field_check.py check site.json 사진1.jpg [사진2.jpg ...]
       -> 사진마다 숫자 표 + A4 판정(2장 이상일 때) + A5 판정(설정에 실측값이 있을 때)
       -> 사진_check.jpg 에 찾은 경계를 그려서 저장(눈으로 맞게 잡았는지 확인용)
"""
import sys
import json
import os
import numpy as np
import cv2


# ---------------------------------------------------------------
# 1. 경계 찾기: 한 줄(세로 단면)에서 어두운 실루엣의 위·아래 경계
# ---------------------------------------------------------------
def edges_in_profile(prof):
    """밝은 배경 - 어두운 물체 - 밝은 배경 으로 된 한 줄에서 물체의 두 경계(소수점 px)를 찾는다."""
    prof = prof.astype(np.float64)
    n = len(prof)
    # 물체 밝기 = 가장 어두운 쪽 10% 의 값
    dark = np.percentile(prof, 10)
    # 1단계: 물체가 대략 어디 있는지 찾는다(양 끝 중 밝은 쪽을 배경으로 보고 반값 문턱)
    k = max(3, n // 10)
    bg0 = max(np.median(prof[:k]), np.median(prof[-k:]))
    if bg0 - dark < 30:
        return None  # 배경과 물체 차이가 작다 = 실루엣이 없다
    below = np.where(prof < (bg0 + dark) / 2.0)[0]
    if len(below) == 0:
        return None
    # 문턱 아래 구간이 여러 덩어리일 수 있다.
    #  - ROI 끝에 닿은 덩어리 = 끝에 걸린 다른 물체 -> 버린다
    #  - 나머지는 처음~끝을 하나로 본다. 물체 앞면 반사 줄이 실루엣을 두 덩어리로 쪼개도
    #    바깥 경계는 그대로 잡히게 (10-01 집 사진: 불 켠 사진에서 반사 줄이 위쪽 경계로 잡혔음)
    cuts = np.where(np.diff(below) > 1)[0] + 1
    runs = [r for r in np.split(below, cuts) if r[0] >= 3 and r[-1] <= n - 4]
    if not runs:
        return None  # 물체가 ROI 끝에 닿음 -> ROI 를 더 넓게 잡아야 함
    below = np.arange(runs[0][0], runs[-1][-1] + 1)
    # 2단계: 배경 밝기를 경계 '바로 바깥'에서 위·아래 따로 잰다.
    # ROI 끝에서 재면 끝에 걸린 다른 물체(10-01 집 사진: 핀 ROI 에 걸린 샤프트 끝)에 문턱이 끌려간다
    w = max(3, min(k, 40))
    bg1 = np.percentile(prof[max(0, below[0] - w):below[0]], 90)
    bg2 = np.percentile(prof[below[-1] + 1:below[-1] + 1 + w], 90)
    if min(bg1, bg2) - dark < 30:
        return None
    t1 = (bg1 + dark) / 2.0
    t2 = (bg2 + dark) / 2.0
    # 3단계: 물체 안쪽에서 바깥으로 걸어 나가며 문턱을 처음 넘는 곳 = 경계
    # (바깥쪽에 다른 어두운 물체가 있어도 그 너머까지 가지 않는다)
    i = below[0]
    while i > 0 and prof[i - 1] < t1:
        i -= 1
    j = below[-1]
    while j < n - 1 and prof[j + 1] < t2:
        j += 1
    if i == 0 or j == n - 1:
        return None
    # 소수점 보정: 문턱을 지나는 두 px 사이를 직선으로 이어 정확한 위치를 구한다
    top = (i - 1) + (prof[i - 1] - t1) / (prof[i - 1] - prof[i])
    bot = j + (prof[j] - t2) / (prof[j] - prof[j + 1])
    # 경계 번짐 폭: 밝기가 배경 쪽 90% 에서 물체 쪽 10% 까지 바뀌는 데 걸리는 px 수 (작을수록 선명)
    def cross(level, start, step, want_above):
        """start 에서 step 방향으로 걸어가며 밝기가 level 을 처음 넘는 소수점 위치."""
        a = start
        while 0 < a < n - 1:
            nxt = a + step
            if (prof[nxt] >= level) == want_above and (prof[a] >= level) != want_above:
                return a + step * (level - prof[a]) / (prof[nxt] - prof[a])
            a = nxt
        return float(a)
    def spread(bg, start, step):
        hi = dark + 0.9 * (bg - dark)
        lo = dark + 0.1 * (bg - dark)
        # 경계 지점에서 배경 쪽(-step)으로 90% 를 넘는 곳, 물체 쪽(+step)으로 10% 아래로 가는 곳
        p_hi = cross(hi, start, -step, True)
        p_lo = cross(lo, start, step, False)
        return abs(p_hi - p_lo)
    e1, e2 = spread(bg1, i, 1), spread(bg2, j, -1)
    return top, bot, min(bg1, bg2), dark, (e1 + e2) / 2.0, e1, e2


def measure_width(gray, roi, axis):
    """ROI 안에서 물체 폭을 줄마다 재서 중앙값을 낸다.
    axis='v' : 세로 폭(샤프트, 눕힌 핀) -> 열(column)마다 세로 단면
    axis='h' : 가로 폭(세운 핀)        -> 행(row)마다 가로 단면
    """
    x0, x1, y0, y1 = roi
    sub = gray[y0:y1, x0:x1]
    if axis == 'h':
        sub = sub.T  # 가로 폭은 돌려서 세로 폭처럼 잰다
    widths, tops, bots, contrast, idx, edges, e1s, e2s = [], [], [], [], [], [], [], []
    for c in range(sub.shape[1]):
        r = edges_in_profile(sub[:, c])
        if r is None:
            continue
        top, bot, bg, dark, edge, e1, e2 = r
        edges.append(edge)
        e1s.append(e1)
        e2s.append(e2)
        widths.append(bot - top)
        tops.append(top)
        bots.append(bot)
        contrast.append(bg - dark)
        idx.append(c)
    if len(widths) < 5:
        return None
    w = np.array(widths)
    return {
        'width_px': float(np.median(w)),
        'sd_px': float(np.std(w)),          # 줄마다 폭이 얼마나 다른가(기울기·흐림·먼지 신호)
        'n_lines': len(w),
        'n_total': sub.shape[1],
        'contrast': float(np.median(contrast)),
        'edge_px': float(np.median(edges)),   # 경계 번짐 폭 중앙값
        'edge1_px': float(np.median(e1s)),    # 위(가로면 왼쪽) 경계 번짐
        'edge2_px': float(np.median(e2s)),    # 아래(가로면 오른쪽) 경계 번짐
        'tops': np.array(tops), 'bots': np.array(bots), 'idx': idx,
        'widths': w,
    }


def pin_unreliable(m):
    """핀 절대 폭을 믿기 어려운지. A4(변화만 봄)는 괜찮지만 A5(절대 크기)는 흔들린다.
    10-01 집 사진: 연마된 핀 옆면이 넓은 백라이트를 반사해 한쪽에 밝은 테두리가 생겼다
    -> 그쪽 경계만 넓게 번지고(17.7 vs 4.8px), 줄마다 폭이 1% 넘게 흔들렸다."""
    lo, hi = sorted([m['edge1_px'], m['edge2_px']])
    asym = hi > 1.5 * lo and hi - lo > 3          # 한쪽만 1.5배 넘게 번짐
    uneven = m['sd_px'] > 0.005 * m['width_px']   # 줄마다 0.5% 넘게 흔들림
    return asym or uneven


def saturation_ratio(gray, roi):
    """배경 ROI 안에서 255 근처(250 이상)로 날아간 px 비율."""
    x0, x1, y0, y1 = roi
    sub = gray[y0:y1, x0:x1]
    return float((sub >= 250).mean())


# ---------------------------------------------------------------
# 2. 원근 겉보기 과대 보정 (계획서 3.5절)
#    가까이서 둥근 물체를 보면 실루엣 지름이 반지름^3 / D^2 만큼 크게 보인다
# ---------------------------------------------------------------
def apparent(d_mm, D_mm):
    r = d_mm / 2.0
    return d_mm + r ** 3 / D_mm ** 2


def _out_path(path, suffix, out_dir):
    """결과 그림 경로. out_dir 를 주면 사진 폴더를 어지럽히지 않고 거기에 모은다."""
    base = os.path.splitext(os.path.basename(path))[0] + suffix
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, base)
    return os.path.splitext(path)[0] + suffix


# ---------------------------------------------------------------
# 3. 좌표 눈금 그림 (ROI 정할 때 쓰는 것)
# ---------------------------------------------------------------
def make_grid(path, out_dir=None):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    s = 2000.0 / max(H, W)  # 화면에서 보기 좋게 긴 변 2000px 로 줄인다
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    step = 500  # 원본 기준 500px 마다 눈금
    for x in range(0, W, step):
        X = int(x * s)
        col = (0, 0, 255) if x % 2000 == 0 else (0, 200, 255)
        cv2.line(small, (X, 0), (X, small.shape[0]), col, 1)
        if x % 1000 == 0:
            cv2.putText(small, str(x), (X + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    for y in range(0, H, step):
        Y = int(y * s)
        col = (0, 0, 255) if y % 2000 == 0 else (0, 200, 255)
        cv2.line(small, (0, Y), (small.shape[1], Y), col, 1)
        if y % 1000 == 0:
            cv2.putText(small, str(y), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    out = _out_path(path, '_grid.jpg', out_dir)
    cv2.imwrite(out, small)
    print(f'원본 크기 {W} x {H}, 눈금 500px 간격 -> {out}')
    return out


# ---------------------------------------------------------------
# 4. 확인 (본 기능)
# ---------------------------------------------------------------
def draw_check(path, gray, cfg, res, out_dir=None):
    """ROI 주변을 원본 해상도로 잘라 찾은 경계를 그린다.
    전체 사진을 줄이면 경계선이 1px 도 안 돼서 안 보이므로, 칸마다 확대 크롭을 옆으로 붙인다."""
    tiles = []
    for key, color in (('shaft', (0, 0, 255)), ('pin', (0, 255, 0))):
        if key not in cfg:
            continue
        x0, x1, y0, y1 = cfg[key]['roi']
        pad = 60
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = cv2.cvtColor(gray[Y0:y1 + pad, X0:x1 + pad], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(crop, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (255, 0, 0), 2)
        m = res.get(key)
        if m is not None:
            axis = cfg[key].get('axis', 'v')
            # 줄마다 찾은 경계 점을 찍는다 -> 점이 실루엣 가장자리에 붙어 있으면 맞게 잡은 것
            for c, (t, b) in enumerate(zip(m['tops'], m['bots'])):
                for e in (t, b):
                    if axis == 'v':
                        pt = (x0 - X0 + m['idx'][c], int(round(y0 - Y0 + e)))
                    else:
                        pt = (int(round(x0 - X0 + e)), y0 - Y0 + m['idx'][c])
                    cv2.circle(crop, pt, 1, color, -1)
        # 크롭 높이를 600px 로 맞춰서 나란히 붙인다
        s = 600.0 / crop.shape[0]
        tiles.append(cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA))
    out = _out_path(path, '_check.jpg', out_dir)
    if tiles:
        cv2.imwrite(out, np.hstack(tiles))
    return out


def check(cfg_path, paths, out_dir=None):
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    rows = []
    for p in paths:
        # 흑백으로 읽는다(경계는 밝기만 있으면 된다). 200MP 도 1~2초면 읽힌다
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f'[읽기 실패] {p}')
            continue
        res = {}
        res['shaft'] = measure_width(gray, cfg['shaft']['roi'], 'v')
        if 'pin' in cfg:
            res['pin'] = measure_width(gray, cfg['pin']['roi'], cfg['pin'].get('axis', 'h'))
        sat = saturation_ratio(gray, cfg['bg']['roi']) if 'bg' in cfg else None
        out = draw_check(p, gray, cfg, res, out_dir)
        rows.append((os.path.basename(p), res, sat, out))

    # --- 사진별 표 ---
    print()
    print(f"{'사진':<24}{'샤프트px':>10}{'(sd)':>7}{'번짐':>6}{'핀px':>10}{'(sd)':>7}{'포화%':>7}{'대비':>6}  경계그림")
    for name, res, sat, out in rows:
        s, q = res.get('shaft'), res.get('pin')
        sv = f"{s['width_px']:10.2f}{s['sd_px']:7.2f}{s['edge_px']:6.1f}" if s else f"{'못찾음':>10}{'':>13}"
        qv = f"{q['width_px']:10.2f}{q['sd_px']:7.2f}" if q else f"{'못찾음':>10}{'':>7}"
        tv = f"{sat*100:7.2f}" if sat is not None else f"{'-':>7}"
        cv = f"{s['contrast']:6.0f}" if s else f"{'-':>6}"
        print(f"{name:<24}{sv}{qv}{tv}{cv}  {os.path.basename(out)}")
        for key in ('shaft', 'pin'):
            m = res.get(key)
            # 경계를 찾은 줄이 전체의 90% 가 안 되면 ROI 안에 가리는 것이 있다는 뜻
            if m and m['n_lines'] < 0.9 * m['n_total']:
                print(f"   ! {key}: 경계를 찾은 줄이 {m['n_lines']}/{m['n_total']} 뿐 -> ROI 안에 받침·그림자·끝부분이 섞였는지 확인")
            # 줄마다 폭이 0.5% 넘게 흔들리면 경계가 일정하지 않다는 뜻
            # (10-01 집 사진: 세운 핀 옆면의 회색 반사 띠 때문에 핀 폭이 줄마다 246~254px -> sd 1.2%)
            if key == 'pin' and m and pin_unreliable(m):
                print(f"   ! pin: 좌우 경계 번짐 {m['edge1_px']:.1f} / {m['edge2_px']:.1f}px -> 테두리(반사)·흐림 의심. A4 는 써도 되지만 A5(절대 크기) 판정은 참고만")
            if m and m['sd_px'] > 0.005 * m['width_px']:
                print(f"   ! {key}: 줄마다 폭 흔들림 {m['sd_px']:.2f}px ({m['sd_px']/m['width_px']*100:.2f}%) -> 경계그림에서 반사 띠·테이퍼 끝·먼지 확인, ROI 를 고른 구간으로 좁히기")

    if 'bg' in cfg:
        for name, res, sat, out in rows:
            if sat is not None and sat > 0.01:
                print(f"   ! {name}: 배경 포화 {sat*100:.1f}% (1% 초과) -> 백라이트 밝기 한 단계 낮추기 (A1)")

    # --- A4: 연속 사진의 핀 px 흔들림 (기준 = 첫 장 또는 설정의 ref_pin_px) ---
    pins = [(n, r['pin']['width_px']) for n, r, _, _ in rows if r.get('pin')]
    if len(pins) >= 2 or (pins and cfg.get('ref_pin_px')):
        ref = cfg.get('ref_pin_px') or pins[0][1]
        print(f"\n[A4 / 멈출 신호] 핀 px 기준 {ref:.2f}px, 허용 0.1%")
        for n, v in pins:
            d = (v - ref) / ref * 100
            flag = '통과' if abs(d) <= 0.1 else '★초과 -> 폰·거치대·렌즈 전환 점검'
            print(f"   {n:<24}{v:10.2f}px  {d:+7.3f}%  {flag}")
        spread = (max(v for _, v in pins) - min(v for _, v in pins)) / ref * 100
        print(f"   최대-최소 차이 {spread:.3f}%")

    # --- 묶음 요약: 여러 장일 때 장간 흔들림을 µm 로 (줌·해상도가 달라도 비교되게) ---
    both = [r for _, r, _, _ in rows if r.get('shaft') and r.get('pin')]
    if len(both) >= 2:
        pin_mm = cfg.get('pin_mm', 10.0)
        um_px = np.mean([pin_mm * 1000 / r['pin']['width_px'] for r in both])   # 핀으로 잰 px 하나의 크기
        sh = np.array([r['shaft']['width_px'] for r in both])
        ed = np.array([r['shaft']['edge_px'] for r in both])
        # 핀으로 환산한 샤프트 지름(장마다) -> 그 흔들림이 '측정값' 흔들림
        dia = np.array([r['shaft']['width_px'] * pin_mm / r['pin']['width_px'] for r in both])
        print(f"\n[묶음 요약] {len(both)}장, px 하나 = {um_px:.1f}µm (핀 기준)")
        print(f"   샤프트 px 장간 sd   {sh.std(ddof=1):.3f}px = {sh.std(ddof=1)*um_px:.2f}µm")
        print(f"   핀 환산 지름 장간 sd {dia.std(ddof=1)*1000:.2f}µm  (평균 {dia.mean():.4f}mm)")
        print(f"   샤프트 경계 번짐   {ed.mean():.1f}px = {ed.mean()*um_px:.0f}µm")

    # --- A5: 핀으로 환산한 샤프트 지름 vs 실측 ---
    true_shaft = cfg.get('true_shaft_mm')
    if true_shaft and pins:
        pin_mm = cfg.get('pin_mm', 10.0)
        D = cfg.get('D_mm', 400.0)
        print(f"\n[A5] 실측 {true_shaft:.3f}mm, 핀 {pin_mm:.3f}mm, D={D:.0f}mm (겉보기 과대 보정 포함), 허용 {cfg.get('a5_tol_pct', 0.1)}%")
        for n, r, _, _ in rows:
            if not (r.get('pin') and r.get('shaft')):
                continue
            mm_per_px = apparent(pin_mm, D) / r['pin']['width_px']      # 핀의 '보이는' 지름으로 배율을 잡는다
            shaft_seen = r['shaft']['width_px'] * mm_per_px              # 샤프트의 '보이는' 지름
            shaft_est = shaft_seen - (true_shaft / 2) ** 3 / D ** 2       # 보이는 지름에서 과대분을 뺀다
            d = (shaft_est - true_shaft) / true_shaft * 100
            tol = cfg.get('a5_tol_pct', 0.1)
            if abs(d) <= tol:
                flag = '통과'
            elif pin_unreliable(r['pin']):
                # 핀 경계가 고르지 않으면 차이가 깊이 때문인지 테두리 때문인지 모른다 -> 핀을 옮기라고 하지 않는다
                flag = '초과(핀 경계 고르지 않음 -> 이동 안내 생략, 캘리퍼스 깊이 맞춤을 믿고 진행·값만 기록)'
            else:
                # 환산이 크다 = 핀이 작게 보인다 = 핀이 샤프트보다 멀다 -> 핀을 카메라 쪽으로
                move = abs(d) / 100 * D
                way = '카메라 쪽(앞)으로' if d > 0 else '뒤로'
                flag = f'★초과 -> 핀을 약 {move:.1f}mm {way}'
            print(f"   {n:<24}환산 {shaft_est:8.4f}mm  {d:+7.3f}%  {flag}")

    # --- 판정: 표를 다 읽지 않아도 되게 한 줄로 ---
    stop, notes = [], []
    lo, hi = cfg.get('shaft_px_range', [350, 650])   # 400mm·200MP 에서 지름 20mm 는 약 500px
    for name, res, sat, out in rows:
        sm, pm = res.get('shaft'), res.get('pin')
        if sm is None:
            stop.append(f'{name}: 샤프트 못 찾음 -> 좌표·차광 확인')
            continue
        if not (lo <= sm['width_px'] <= hi):
            stop.append(f'{name}: 샤프트 {sm["width_px"]:.0f}px 가 예상({lo}~{hi})과 다름 -> 좌표가 다른 물체에 잡혔거나 거리가 다름')
        if sm['n_lines'] < 0.9 * sm['n_total'] or sm['sd_px'] > 0.005 * sm['width_px']:
            stop.append(f'{name}: 샤프트 경계 고르지 않음 -> 차광·초점 확인')
        if sat is not None and sat > 0.01:
            stop.append(f'{name}: 배경 포화 {sat*100:.1f}% -> 백라이트 밝기 낮추기')
        if pm is None:
            notes.append(f'{name}: 핀 못 찾음 (폰 움직임 감시 불가)')
        elif pin_unreliable(pm):
            notes.append('핀 테두리 경고 -> A5 는 참고만')
    if len(pins) >= 2 or (pins and cfg.get('ref_pin_px')):
        ref = cfg.get('ref_pin_px') or pins[0][1]
        moved = [n for n, v in pins if abs(v - ref) / ref * 100 > 0.1]
        if moved:
            stop.append(f'핀 px 0.1% 넘게 변함({", ".join(moved)}) -> 폰·거치대·초점 점검')
    notes = list(dict.fromkeys(notes))   # 같은 말 한 번만
    ok = not stop
    print("\n==============================")
    print("판정: " + ("진행 OK" if ok else "다시 세팅"))
    for m in stop:
        print("  X " + m)
    for m in notes:
        print("  - " + m)
    print("==============================")
    return {'ok': ok, 'stop': stop, 'notes': notes}

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'grid':
        make_grid(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == 'check':
        check(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__)
