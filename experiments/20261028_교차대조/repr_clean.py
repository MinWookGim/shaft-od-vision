# ② 대표값 후보(Median·Mean·P90·P95·상위3·5·Max·envelope)를 공통 규칙(LOSO 9, 시편 중앙값)과
#    교정 방식 4가지에서 다시 비교. 열은 31개 전부 vs 시편 공통으로 튀는 열을 뺀 21개(8069~8262)
import numpy as np, pandas as pd
TRUE = {2: 20.020, 3: 20.028, 4: 20.019, 5: 20.030, 6: 20.011, 7: 20.014, 8: 20.026, 9: 20.009, 10: 20.022, 11: 19.968, 12: 20.044}
NORMAL = list(range(2, 11))
rc = pd.read_csv('../1027_repr/repr_cols.csv'); rc['no'] = rc.no.str[2:].astype(int)
pin = pd.read_csv('../1021_main/local_edges.csv')[['file', 'pin']]
rc = rc.merge(pin, on='file'); rc = rc[rc.no != 1].reset_index(drop=True)
XS = np.linspace(8030, 8320, 31).round().astype(int)
WIN = {'31열': list(range(31)), '21열': [i for i, x in enumerate(XS) if 8060 <= x <= 8270]}
T0 = rc[[f't{i}' for i in range(31)]].to_numpy(); B0 = rc[[f'b{i}' for i in range(31)]].to_numpy()
def topk(a, k): a = a[~np.isnan(a)]; return np.sort(a)[-k:].mean()
C = {'Median': lambda D, T, B: np.nanmedian(D), 'Mean': lambda D, T, B: np.nanmean(D),
     'P90': lambda D, T, B: np.nanpercentile(D, 90), 'P95': lambda D, T, B: np.nanpercentile(D, 95),
     '상위3': lambda D, T, B: topk(D, 3), '상위5': lambda D, T, B: topk(D, 5),
     'Max': lambda D, T, B: np.nanmax(D), 'envelope': lambda D, T, B: np.nanmax(B) - np.nanmin(T)}
pin_ref = rc.pin.median(); s = 16 / pin_ref
rows = []
for w, idx in WIN.items():
    T, B = T0[:, idx], B0[:, idx]; D = B - T
    for k, f in C.items():
        px = pd.Series([f(D[i], T[i], B[i]) for i in range(len(rc))])
        V = {'3 물리': (px, s, False), '3a 물리+핀1:1': (px + pin_ref - rc.pin, s, False),
             '5 비율x16': (px / rc.pin, 16.0, False), '4 비율 선형(자유)': (px / rc.pin, None, True)}
        for mname, (x, a_phys, free) in V.items():
            sp = x.groupby(rc.no).median(); y = pd.Series(TRUE)
            e = []
            for h in NORMAL:
                tr = [n for n in NORMAL if n != h]
                if free: a, b = np.polyfit(sp[tr], y[tr], 1)
                else: a = a_phys; b = (y[tr] - a * sp[tr]).mean()
                e.append((a * sp[h] + b - y[h]) * 1000)
            sd = (x * (a_phys or 16 / pin_ref * 1)).groupby(rc.no[rc.stage == 1]).std().median() * 1000 if not free else np.nan
            rows.append(dict(창=w, 대표값=k, 방식=mname, MAE=round(np.abs(e).mean(), 1)))
r = pd.DataFrame(rows); r.to_csv('repr_clean.csv', index=False)
pd.set_option('display.width', 200)
for m in r.방식.unique():
    print(f'\n[{m}]  (평균 부르기 6.64µm)')
    print(r[r.방식 == m].pivot(index='대표값', columns='창', values='MAE').loc[list(C)].to_string())
