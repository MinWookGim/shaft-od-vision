# 카메라 상태·초점 흐름을 감안한 재분석 (2026-10-11)
# 재료: 핀 B(핀 왼쪽 끝 기준 s 60~105, 1011_pincol), 샤프트 M0(1008_methods), 카메라 상태(state.csv: 받침대 경계·창 아래 경계)
# 규칙(돌리기 전에 고정)
#   1) 묶음 안 흐름: 찍은 순서에 대한 기울기(핀 px, 창 아래 경계 y, 지름). 같은 묶음 10장 안에서 초점·카메라가 흘렀나
#   2) 같은 상태 쌍: 카메라 상태가 거의 같고 실측도 같은 두 시편(NO3·NO4 = 20.030, NO9·NO10 = 20.015)의 측정 차이
#      -> 상태가 같으면 측정이 얼마나 맞나 (교정 없이도 볼 수 있다: 같은 실측이니 차이가 곧 오차)
#   3) 흐름 빼기: 묶음마다 마지막 5장만 쓰면(초점이 자리 잡은 뒤) LOO 가 어떻게 바뀌나
import os, sys, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '1011_pincol'))
z = np.load(os.path.join(HERE, '..', '1011_pincol', '_out', 'pin_scan.npz')); XS, W, FILES = z['XS'], z['BOT'] - z['TOP'], list(z['files'])
def pinB(k):
    w = W[k]; ok = np.where((w > 250) & (XS >= 9500))[0]; xL = XS[ok[0]]; s = XS - xL
    return np.nanmedian(w[(s >= 60) & (s <= 105)])
PB = {f: pinB(k) for k, f in enumerate(FILES)}
SH = {r['파일']: float(r['지름px']) for r in csv.DictReader(open(os.path.join(HERE, '..', '1008_methods', '_out', 'methods_per_image.csv'), encoding='utf-8-sig')) if r['방법'] == 'M0_50%경계(기존)'}
ST = list(csv.DictReader(open(os.path.join(HERE, '_out', 'state.csv'), encoding='utf-8-sig')))
TRUE = {int(r['묶음']): float(r['실측mm']) for r in csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig'))}
NAME = {int(r['묶음']): r['시편'] for r in ST}
NORMAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13]; NG = [11, 12]
G = {}
for r in ST: G.setdefault(int(r['묶음']), []).append(r)
out = []
def P(s=''): print(s); out.append(s)

P('== 1) 묶음 안 흐름 (찍은 순서 1~10 에 대한 기울기, 10장 전체 변화량 = 기울기 x 9)')
for g in sorted(G):
    rs = G[g]; i = np.arange(len(rs))
    pin = np.array([PB[r['파일']] for r in rs]); yB = np.array([float(r['yB']) for r in rs])
    d = np.array([SH[r['파일']] / PB[r['파일']] * 10 for r in rs]) * 1000
    sl = lambda v: np.polyfit(i, v, 1)[0] * 9
    P(f"묶음{g:2d} {NAME[g]:4s} {rs[0]['초점']}  핀 {sl(pin):+.2f}px  창아래 {sl(yB):+.2f}px  지름 {sl(d):+6.1f}um  (지름 10장 sd {d.std(ddof=1):4.1f}um)")

def est(last=None):
    e = {}
    for g in G:
        rs = G[g] if last is None else G[g][-last:]
        e[g] = np.mean([SH[r['파일']] / PB[r['파일']] * 10 for r in rs])
    return e
P('\n== 2) 같은 상태·같은 실측 쌍 (교정 없이 두 측정값의 차이 = 오차의 차이)')
e = est()
for a, b in ((3, 4), (9, 10)):
    sa, sb = G[a][0], G[b][0]
    P(f"{NAME[a]} vs {NAME[b]} (실측 {TRUE[a]:.3f} / {TRUE[b]:.3f})  측정 차이 {(e[a] - e[b]) * 1000:+.1f}um   "
      f"상태 차이: 받침대 경계 {float(sb['xL']) - float(sa['xL']):+.1f}px, 간격 {float(sb['간격']) - float(sa['간격']):+.1f}px, 창아래 {float(sb['yB']) - float(sa['yB']):+.1f}px")
P('비교: 상태가 다른 같은 실측 쌍')
for a, b in ((2, 5), (7, 8), (13, 6)):
    P(f"{NAME[a]} vs {NAME[b]} (실측 {TRUE[a]:.3f} / {TRUE[b]:.3f})  측정차-실측차 {((e[a] - e[b]) - (TRUE[a] - TRUE[b])) * 1000:+.1f}um")

P('\n== 3) LOO (정상 10개, 평균 찍기 5.8um)')
for lab, last in (('10장 전부', None), ('마지막 5장만', 5)):
    e = est(last); loo = []
    for g in NORMAL:
        off = np.mean([e[h] - TRUE[h] for h in NORMAL if h != g]); loo.append((e[g] - off - TRUE[g]) * 1000)
    off = np.mean([e[h] - TRUE[h] for h in NORMAL])
    P(f"{lab}: MAE {np.mean(np.abs(loo)):.1f} 최대 {np.max(np.abs(loo)):.1f} | NG {(e[11] - off - TRUE[11]) * 1000:+.0f}, {(e[12] - off - TRUE[12]) * 1000:+.0f} | "
      + ' '.join(f'{NAME[g]} {v:+.0f}' for g, v in zip(NORMAL, loo)))
open(os.path.join(HERE, '_out', 'reanalyze.txt'), 'w').write('\n'.join(out))
