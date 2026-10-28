# ① 같은 3차 사진인데 최선 MAE 가 3.99(팀원 방식4) 대 8.6µm(내 시간 보정)로 갈린 이유 가르기
#   경계 쪽 차이(어떤 경계 값을 쓰나)와 교정 쪽 차이(어떤 식으로 mm 로 바꾸나)를 표 하나에 엇갈려 놓는다
# ② 창 점검: 내 31개 열 중 시편 공통으로 튀는 열(8049 아래 굴곡, 8281 위 스파이크, 8310·8320 단 끝)을 뺀 21개 열 판
# 공통 규칙(팀원 E6 사전 규칙과 같게): NO1 제외, 정상 9개(NO2~NO10, NO9 포함) 하나씩 빼고 맞히기(LOSO),
#   시편 대표값 = 그 시편 사진(본시험+재장착) 지름의 중앙값, MAE 는 9개 평균
import numpy as np, pandas as pd, sys
GON = sys.argv[1]
TRUE = {2: 20.020, 3: 20.028, 4: 20.019, 5: 20.030, 6: 20.011, 7: 20.014, 8: 20.026, 9: 20.009, 10: 20.022, 11: 19.968, 12: 20.044}
NORMAL = list(range(2, 11))

# 경계 원천 4가지 -> 사진마다 샤프트 px, 핀 px
g = pd.read_csv(GON, encoding='utf-8-sig')
g['no'] = g.specimen.str[2:].astype(int)
loc = pd.read_csv('../1021_main/local_edges.csv'); loc['no'] = loc.no.str[2:].astype(int)
rc = pd.read_csv('../1027_repr/repr_cols.csv')
XS = np.linspace(8030, 8320, 31).round().astype(int)
CLEAN = [i for i, x in enumerate(XS) if 8060 <= x <= 8270]
T = rc[[f't{i}' for i in range(31)]].to_numpy(); B = rc[[f'b{i}' for i in range(31)]].to_numpy(); D = B - T
rc['mean31'] = np.nanmean(D, 1); rc['mean21'] = np.nanmean(D[:, CLEAN], 1)
rc['med31'] = np.nanmedian(D, 1); rc['med21'] = np.nanmedian(D[:, CLEAN], 1)
base = loc[['file', 'stage', 'no', 'pin', 'shaft']].merge(rc[['file', 'mean31', 'mean21', 'med31', 'med21']], on='file')
base = base.merge(g[['filename', 'bearing_cross50_px', 'pin_cross50_px']].rename(columns={'filename': 'file'}), on='file')
base = base[base.no != 1]
SRC = {
    '팀원 cross50 (약 110열 강건 직선)': ('bearing_cross50_px', 'pin_cross50_px'),
    '내 국소50% 32열 중앙값 (3차 분석정리)': ('shaft', 'pin'),
    '내 국소50% 31열 평균 (대표값 비교)': ('mean31', 'pin'),
    '내 국소50% 21열 평균 (튀는 열 뺌)': ('mean21', 'pin'),
}

def fit_eval(spec, xcol, kind):
    # kind: const / lin / phys(기울기 = 물리 배율 고정, 절편만) ; xcol 은 spec 의 열 이름
    def fit(tr):
        y = tr.true.values; x = tr[xcol].values
        if kind == 'const': return lambda v: y.mean() + 0 * v, 0.0
        if kind == 'lin': a, b = np.polyfit(x, y, 1); return lambda v: a * v + b, a
        a = PHYS[xcol]; b = (y - a * x).mean(); return lambda v: a * v + b, a
    e = []
    for h in NORMAL:
        f, _ = fit(spec.loc[[n for n in NORMAL if n != h]]); e.append((f(spec.loc[h, xcol]) - spec.loc[h, 'true']) * 1000)
    f, a = fit(spec.loc[NORMAL])
    return np.abs(e).mean(), np.array(e), a, f(spec.loc[11, xcol]), f(spec.loc[12, xcol])

rows = []
for sel in ['all', 'burst']:
    b0 = base if sel == 'all' else base[base.stage == 1]
    for sname, (sc, pc) in SRC.items():
        d = b0[['no', 'stage', sc, pc]].rename(columns={sc: 'px', pc: 'pin'}).copy()
        pin_ref = d.pin.median()
        d['px_add1'] = d.px + (pin_ref - d.pin)                  # 내 시간 보정: 핀이 잃은 px 를 그대로 더함(1:1)
        d['px_add048'] = d.px + 0.478 * (pin_ref - d.pin)        # 팀원 비율 실험 기울기 0.478 로 더함(탐색)
        d['ratio'] = d.px / d.pin
        spec = d.groupby('no')[['px', 'px_add1', 'px_add048', 'ratio']].median()
        spec['true'] = pd.Series(TRUE)
        s = 16 / pin_ref
        PHYS = {'px': s, 'px_add1': s, 'px_add048': s, 'ratio': 16.0}
        M = [('0 평균 부르기', 'px', 'const'), ('2 전체 선형(기울기 자유)', 'px', 'lin'), ('3 물리 배율(V1)', 'px', 'phys'),
             ('3a 물리 + 핀 px 1:1 더함(내 보정)', 'px_add1', 'phys'), ('3b 물리 + 핀 px x0.478 더함', 'px_add048', 'phys'),
             ('4 핀 비율 선형(기울기 자유)', 'ratio', 'lin'), ('5 핀 비율 x16 고정', 'ratio', 'phys')]
        for mname, xc, kind in M:
            mae, e, a, ng1, ng2 = fit_eval(spec, xc, kind)
            slope_rel = (a / PHYS[xc]) if kind == 'lin' else (1.0 if kind == 'phys' else 0.0)
            rows.append(dict(사진=sel, 경계=sname, 방식=mname, MAE=round(mae, 2), 최대=round(np.abs(e).max(), 1),
                             기울기_물리대비=round(slope_rel, 2), NG1=round(ng1, 4), NG2=round(ng2, 4),
                             시편오차=' '.join(f'{v:+.0f}' for v in e)))
r = pd.DataFrame(rows); r.to_csv('cross_calib.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
for sel in ['all', 'burst']:
    print(f'\n===== 사진 {sel} =====')
    print(r[r.사진 == sel].pivot(index='방식', columns='경계', values='MAE').to_string())
print('\n기울기(물리 대비, 기울기 자유 방식만)')
print(r[(r.사진 == 'all') & r.방식.str.contains('자유')][['경계', '방식', '기울기_물리대비', 'NG1', 'NG2']].to_string(index=False))
print('\n시편별 오차(all, µm, NO2..NO10)')
print(r[(r.사진 == 'all') & r.방식.str.match('(3|4|5)')][['경계', '방식', '시편오차']].to_string(index=False))
