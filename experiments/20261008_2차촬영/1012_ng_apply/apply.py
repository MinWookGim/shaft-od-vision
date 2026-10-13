# 선행 개발자 10-10 보고서 교정식을 10-08 우리 측정값(NG1, NG2 포함)에 적용해 본다
# 교정식(보고서 15절): mm = 0.002573527738 x px + 18.614463952718  (NO4~6, 3점)
# 판정: 제품 규격 20.010~20.030 이면 OK
# 1) 선행 개발자 식을 그대로: 우리 px 정의가 선행 개발자 px 와 같다고 보고 적용 (px 차이가 크면 해석 주의)
# 2) 선행 개발자 절차를 그대로: 우리 px 로 NO4~6 3점 직선을 새로 맞춰 적용 (기울기도 자유)
# 3) 비교: 배율 물리값 고정(핀 비율) + 치우침만 교정
import csv, numpy as np
A, B = 0.002573527738, 18.614463952718
rows = list(csv.DictReader(open('../1008_field/_out/full_rows.csv', encoding='utf-8-sig')))
eng = list(csv.DictReader(open('../1008_engine/_out/engine_rows_tight.csv', encoding='utf-8-sig')))
true = {int(r['묶음']): float(r['실측mm']) for r in rows}
name = {int(r['묶음']): r['시편'] for r in rows}
def groups(rs, key, flt=lambda r: True):
    g = {}
    for r in rs:
        if flt(r): g.setdefault(int(r['묶음']), []).append(float(r[key]))
    return g
sets = {'M 50%경계(우리)': (groups(rows, '샤프트px'), groups(rows, '핀px')),
        'E 선행엔진 원본규칙': (groups(eng, '샤프트px', lambda r: r['방법'].startswith('E_')), groups(eng, '핀px', lambda r: r['방법'].startswith('E_'))),
        'F 선행엔진 가장센쌍': (groups(eng, '샤프트px', lambda r: r['방법'].startswith('F_')), groups(eng, '핀px', lambda r: r['방법'].startswith('F_')))}
CAL, VAL, NG = [4, 5, 6], [7, 8, 9, 10], [11, 12]
ok = lambda v: 'OK' if 20.010 <= v <= 20.030 else 'NG'
print('선행 개발자 보고서 px: NO6 543.26  NO5 544.86  NO4 550.33')
for nm, (sh, pin) in sets.items():
    m = {g: np.mean(v) for g, v in sh.items()}
    print(f'\n== {nm}: 우리 px NO4 {m[4]:.2f} NO5 {m[5]:.2f} NO6 {m[6]:.2f}')
    a, b = np.polyfit([m[g] for g in CAL], [true[g] for g in CAL], 1)
    # 3) 핀 비율로 배율 고정, NO4~6 으로 치우침만
    est = {g: np.mean(np.array(sh[g]) / np.array(pin[g]) * 10) for g in sh}
    off = np.mean([est[g] - true[g] for g in CAL])
    print(f'   2) 우리 px로 3점 다시 맞춘 기울기 {a * 1000:.2f}um/px (물리 약 36.8)')
    for g in VAL + NG:
        v1 = A * m[g] + B; v2 = a * m[g] + b; v3 = est[g] - off
        # 사진 한 장씩 판정도 센다 (식 1)
        per = [ok(A * x + B) for x in sh[g]]
        print(f'   {name[g]:5s} 실측 {true[g]:.3f}({ok(true[g])}) | 1) 선행식 {v1:.4f} {ok(v1)} [장별 OK {per.count("OK")}/{len(per)}] '
              f'| 2) 3점재적합 {v2:.4f} {ok(v2)} | 3) 핀고정 {v3:.4f} {ok(v3)}')
# 상수 20.020 출력
print('\n상수 20.020: NG1, NG2 모두 OK')
