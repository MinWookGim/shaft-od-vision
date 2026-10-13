# inRange·전부 변형에서 정상 검증 시편은 어떻게 나왔나 (NG 를 NG 로 본 게 측정 덕인지 확인)
import numpy as np, seven as S
for name in ('1 inRange', '전부'):
    f = S.VARS[name]; sh = {}
    for r in S.rows: sh.setdefault(int(r['묶음']), []).append(S.meas(f(S.D[r['파일']]['shaft'])))
    m = {g: np.nanmean(v) for g, v in sh.items()}
    a, b = np.polyfit([m[g] for g in S.CAL], [S.true[g] for g in S.CAL], 1)
    print(name, ' '.join(f"{S.rows[[int(r['묶음']) for r in S.rows].index(g)]['시편']}:px{m[g]:.1f}->{a*m[g]+b:.4f}{S.ok(a*m[g]+b)}(실측{S.true[g]})" for g in S.VAL + S.NG))
