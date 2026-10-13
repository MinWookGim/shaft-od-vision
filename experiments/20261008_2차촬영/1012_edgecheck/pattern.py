# 멘토 조언 2 (10-10): "경계 오차에 패턴이 있으면 규칙을 더하고, 없으면 학습으로 가라"
# 질문: 시편마다 남는 지름 오차(M0, 핀 비율)가 사진에서 잴 수 있는 어떤 값과 함께 움직이나?
#       함께 움직이면 그 값으로 보정하는 '규칙'을 만들 수 있다
# ---- 결과 보기 전에 정한 규칙 ----
#  - 지름 = M0 지름px / 핀px x 10 (1008_methods 와 같음). 시편마다 10장 평균
#  - 정상 시편 = 묶음 2~10, 13 (10개). 묶음 1 제외 (1008_methods 와 같은 이유)
#  - 후보 값(사진마다 재서 시편 평균), 아래 7개로 고정. 결과 보고 추가·삭제 안 함
#      A 아래 단 차이   : M2(바깥 단) - M0 아래 경계 (px). 50% 문턱이 어느 단에 걸렸나
#      B 샤프트 안 밝기 : 아래 경계 15~30px 안쪽 밝기 / 배경. 연한 띠(선반) 높이
#      C 아래 흐림      : 아래 경계 10->90% 폭 (px)
#      D 위 흐림        : 위 경계 10->90% 폭 (px). 초점 상태
#      E 앉은 높이      : 샤프트 중심 y - 핀 중심 y (px)
#      F 배경 밝기
#      G 초점 구간      : F1/F2/F3 별 평균 치우침 따로 빼기
#  - 규칙 시험 = 하나씩 빼고(LOO): 나머지 9개로 '오차 = a + b x 값' 을 맞춰 뺀 시편의 오차를 예측해 뺀다
#  - 비교 잣대: 치우침만 빼기(=기존 M0 LOO 15.5um), 평균 찍기 5.8um
#  - 후보가 7개라 하나쯤은 우연히 좋아 보일 수 있다 -> 평균 찍기를 '넉넉히' 이겨야 규칙 후보로 본다
import os, sys, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '1008_methods'))
import methods as M

rows = M.rows
full = {r['파일']: r for r in csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig'))}
NORMAL, NG = M.NORMAL, M.NG

per = {}                                                 # 묶음 -> 사진별 [지름mm, A..F]
for r in rows:
    rec = M.DATA[r['파일']]; sh = rec['shaft'].astype(float); bg = rec['bg']
    d, p, _ = M.measure(rec, M.m0_half)
    A, B = [], []
    for x in range(0, sh.shape[1], 2):
        col = sh[:, x]
        t0, b0 = M.m0_half(col, bg, M.S_LO, M.S_HI)
        t2, b2 = M.m2_outer(col, bg, M.S_LO, M.S_HI)
        if np.isnan(b0) or np.isnan(b2): continue
        A.append(b2 - b0); B.append(col[int(b0) - 30:int(b0) - 15].mean() / bg)
    f = full[r['파일']]
    E = float(f['샤프트중심y']) - float(f['핀중심y'])
    per.setdefault(int(r['묶음']), []).append([d / p * 10, np.median(A), np.median(B), float(f['흐림아래']), float(f['흐림위']), E, bg])

true = {int(r['묶음']): float(r['실측mm']) for r in rows}
focus = {int(r['묶음']): r['초점'] for r in rows}
spec = {int(r['묶음']): r['시편'] for r in rows}
mean = {g: np.mean(v, 0) for g, v in per.items()}
err = {g: (mean[g][0] - true[g]) * 1000 for g in mean}   # 치우침 빼기 전 오차 um
NAMES = ['A 아래 단 차이(px)', 'B 샤프트 안 밝기', 'C 아래 흐림(px)', 'D 위 흐림(px)', 'E 앉은 높이(px)', 'F 배경 밝기']

def loo_rule(k):
    """k 번째 값으로 직선 보정. k=None 이면 치우침만 빼기"""
    out = {}
    for g in NORMAL:
        tr = [h for h in NORMAL if h != g]
        y = np.array([err[h] for h in tr])
        if k is None: pred = y.mean()
        else:
            x = np.array([mean[h][k + 1] for h in tr]); a, b = np.polyfit(x, y, 1)[::-1]
            pred = a + b * mean[g][k + 1]
        out[g] = err[g] - pred
    return out

def loo_focus():
    out = {}
    for g in NORMAL:
        tr = [h for h in NORMAL if h != g and focus[h] == focus[g]]
        out[g] = err[g] - np.mean([err[h] for h in tr])
    return out

def r_with(k):
    return np.corrcoef([mean[g][k + 1] for g in NORMAL], [err[g] for g in NORMAL])[0, 1]

base = [(np.mean([true[h] for h in NORMAL if h != g]) - true[g]) * 1000 for g in NORMAL]
res = [['(기준) 평균 찍기', '', f'{np.abs(base).mean():.1f}', f'{np.abs(base).max():.1f}', ' '.join(f'{v:+.0f}' for v in base)]]
o = loo_rule(None); e = np.array(list(o.values()))
res.append(['치우침만 빼기 (기존 M0)', '', f'{np.abs(e).mean():.1f}', f'{np.abs(e).max():.1f}', ' '.join(f'{o[g]:+.0f}' for g in NORMAL)])
for k, n in enumerate(NAMES):
    o = loo_rule(k); e = np.array(list(o.values()))
    res.append([n, f'{r_with(k):+.2f}', f'{np.abs(e).mean():.1f}', f'{np.abs(e).max():.1f}', ' '.join(f'{o[g]:+.0f}' for g in NORMAL)])
o = loo_focus(); e = np.array(list(o.values()))
res.append(['G 초점 구간별 치우침', '', f'{np.abs(e).mean():.1f}', f'{np.abs(e).max():.1f}', ' '.join(f'{o[g]:+.0f}' for g in NORMAL)])

hdr = ['보정 규칙', '오차와 상관 r', 'LOO MAE um', '최대 um', '시편별 (' + ' '.join(spec[g] for g in NORMAL) + ')']
w = csv.writer(open(os.path.join(HERE, '_out', 'pattern_result.csv'), 'w', newline='', encoding='utf-8-sig'))
w.writerow(hdr); w.writerows(res)
w2 = csv.writer(open(os.path.join(HERE, '_out', 'pattern_features.csv'), 'w', newline='', encoding='utf-8-sig'))
w2.writerow(['묶음', '시편', '초점', '실측mm', '오차um(치우침전)'] + NAMES)
for g in sorted(mean): w2.writerow([g, spec[g], focus[g], true[g], f'{err[g]:+.1f}'] + [f'{v:.3f}' for v in mean[g][1:]])
for row in res: print(' | '.join(row))
print(); [print(g, spec[g], focus[g], f'{err[g]:+.0f}', ' '.join(f'{v:.2f}' for v in mean[g][1:])) for g in sorted(mean)]
