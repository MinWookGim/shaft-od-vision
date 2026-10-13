# 선행 개발자 "코드 수정 방향 7가지"를 전처리로 구현해 선행 엔진(F: 가장 센 쌍, 박스 안 넓힘 = 보고서 px 재현 경로)에 넣는다
# 7가지 원문: 1 inRange 밝은 띠 제거 / 2 OPEN·CLOSE 좁은 띠 제거 / 3 TOPHAT 밝은 띠 / 4 BLACKHAT 어두운 띠
#            5 connectedComponents 띠 영역 제거 / 6 크기·두께 조건 / 7 ROI 중앙 반사띠 mask
# 원문에 커널 크기·문턱이 없어서 내가 정했다(아래 상수). 결과 보기 전에 고정
# 평가 (결과 보기 전에 고정)
#   a) 선행 절차: NO4~6 3점 직선 -> NO7~10 MAE, NG1·NG2 판정(규격 20.010~20.030)
#   b) 같은 나누기 평균 찍기(20.020) MAE 3.5
#   c) 핀 비율 배율 고정 + 정상 10개 LOO (평균 찍기 5.8 과 비교)
import os, sys, csv, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('SHAFT_CORE_DIR', '.'))   # 선행 엔진 shaft_core.py 가 있는 폴더
import shaft_core as sc
P = sc.Params(vertical_margin=0.0, dia_basis='box', edge_select='strongest')   # F 설정
D = np.load(os.path.join(HERE, 'crops.npz'), allow_pickle=True)['data'][0]
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))
true = {int(r['묶음']): float(r['실측mm']) for r in rows}
K = 25            # 세로 커널 길이(px): 이보다 얇은 띠를 '띠'로 본다 (아래 회색 선반 약 40px 은 안 건드림)
BAND_T = 25       # TOPHAT/BLACKHAT 문턱(밝기)
ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))

def rough(g):
    # 샤프트(어두운 덩어리) 대략 위·아래: 오츠 이진화 후 열마다 어두운 구간의 중앙값
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rr = np.where(b.mean(1) > 127)[0]
    return (rr[0], rr[-1]) if len(rr) else (0, g.shape[0] - 1)

def f1_inrange(g):
    # 1: 샤프트 안쪽(경계에서 10px 들어간 곳)에서 어두운 값보다 확 밝은 화소를 어두운 값으로
    t, b = rough(g); o = g.copy(); a, z = t + 10, b - 10
    if z <= a: return o
    seg = o[a:z]; dark = np.percentile(seg, 10)
    m = cv2.inRange(seg, int(dark + 30), 255) > 0; seg[m] = dark; return o
def f2_openclose(g): return cv2.morphologyEx(cv2.morphologyEx(g, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)
def f3_tophat(g):    return cv2.subtract(g, cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker))      # = OPEN
def f4_blackhat(g):  return cv2.add(g, cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker))         # = CLOSE
def f56_components(g):
    # 5·6: TOPHAT 이 문턱 넘는 밝은 띠를 덩어리로 묶고, 가로로 길고(폭 30% 이상) 얇은(높이 K 이하) 것만 OPEN 값으로 바꾼다
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker); m = (th > BAND_T).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    op = cv2.morphologyEx(g, cv2.MORPH_OPEN, ker); o = g.copy()
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w >= 0.3 * g.shape[1] and h <= K: o[lab == i] = op[lab == i]
    return o
def f7_center(g):
    # 7: 샤프트 지름의 가운데 25~75% 줄을 어두운 값으로 덮는다
    t, b = rough(g); o = g.copy(); d = b - t
    a, z = int(t + .25 * d), int(t + .75 * d)
    if z > a: o[a:z] = np.percentile(o[a:z], 10)
    return o
def f_all(g): return f7_center(f2_openclose(f56_components(f1_inrange(g))))
VARS = {'0 원본(F)': lambda g: g, '1 inRange': f1_inrange, '2 OPEN+CLOSE': f2_openclose, '3 TOPHAT': f3_tophat,
        '4 BLACKHAT': f4_blackhat, '5+6 덩어리·크기': f56_components, '7 중앙 mask': f7_center, '전부': f_all}

def meas(g):
    r = sc.measure_roi(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (0, 0, g.shape[1], g.shape[0]), P)
    return r['measured_pixel'] if r.get('ok') else np.nan

CAL, VAL, NG, NORMAL = [4, 5, 6], [7, 8, 9, 10], [11, 12], [2, 3, 4, 5, 6, 7, 8, 9, 10, 13]
ok = lambda v: 'OK' if 20.010 <= v <= 20.030 else 'NG'
out = []
def Pr(s): print(s, flush=True); out.append(s)
Pr('평균 찍기: 선행 나누기 MAE 3.5um / 정상10 LOO 5.8um')
if __name__ == "__main__":
  for name, f in VARS.items():
      sh, pn = {}, {}
      for r in rows:
          d = D[r['파일']]; g = int(r['묶음'])
          sh.setdefault(g, []).append(meas(f(d['shaft']))); pn.setdefault(g, []).append(meas(f(d['pin'])))
      m = {g: np.nanmean(v) for g, v in sh.items()}
      a, b = np.polyfit([m[g] for g in CAL], [true[g] for g in CAL], 1)
      val = [a * m[g] + b for g in VAL]; mae = np.mean([abs(v - true[g]) for v, g in zip(val, VAL)]) * 1000
      ngv = [a * m[g] + b for g in NG]
      # 장별 NG 판정 수
      ngok = sum(ok(a * x + b) == 'OK' for g in NG for x in sh[g] if not np.isnan(x))
      est = {g: np.nanmean(np.array(sh[g]) / np.array(pn[g]) * 10) for g in sh}
      loo = [(est[g] - np.mean([est[h] - true[h] for h in NORMAL if h != g]) - true[g]) * 1000 for g in NORMAL]
      off = np.mean([est[h] - true[h] for h in NORMAL]); ngp = [(est[g] - off - true[g]) * 1000 for g in NG]
      nan = sum(np.isnan(x) for v in sh.values() for x in v)
      Pr(f"{name:14s} px NO4/5/6 {m[4]:.2f}/{m[5]:.2f}/{m[6]:.2f} 기울기 {a*1000:6.2f}um/px | a) 검증 MAE {mae:4.1f} NG1 {ngv[0]:.4f}{ok(ngv[0])} NG2 {ngv[1]:.4f}{ok(ngv[1])} (NG 20장 중 OK {ngok}) "
         f"| c) 핀고정 LOO MAE {np.mean(np.abs(loo)):5.1f} 최대 {np.max(np.abs(loo)):5.1f} NG오차 {ngp[0]:+.0f},{ngp[1]:+.0f} | 실패 {nan}")
  open(os.path.join(HERE, 'result.txt'), 'w').write('\n'.join(out))
