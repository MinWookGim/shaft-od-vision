# 시간 변화를 뺐을 때 남는 오차 = 다음 촬영으로 줄일 수 있는 한계 추정 (사후 진단, 규칙 밖)
# 경계 깎임은 같은 px 라고 보고(pp_check 결과) 샤프트에 '핀이 잃은 px'를 더해 준다: x_corr = x - (pin - PIN_REF)
import sys, numpy as np, pandas as pd
exec(open('analyze.py').read().split("m['x'] = m[COL]")[0].replace("COL = sys.argv[1] if len(sys.argv) > 1 else 'v3_px'", "COL = 'p50_px'"))
m = m[m.stage != 'noblack'].sort_values('file').copy()
PIN_REF = m.pin_px.median()
UM = 20.02 / 750 * 1000
for name, corr in [('보정 없음', m.p50_px), ('같은 px 보정', m.p50_px - (m.pin_px - PIN_REF)), ('비율 보정', m.p50_px * PIN_REF / m.pin_px)]:
    m['xc'] = corr
    m['mount'] = np.where(m.stage == 'remount', m.groupby(['stage', 'no']).cumcount() + 1, 0)
    mt = m.groupby(['no', 'stage', 'mount']).xc.mean().reset_index()
    s = mt.groupby('no').xc.mean()
    t = pd.Series(TRUE)
    # (a) 물리 배율 고정 + CAL 3 평균으로 치우침만 맞춤 (현실적 교정)
    off = (t[[1,2,3]] - s[[1,2,3]] * UM / 1000).mean()
    pred = off + s * UM / 1000; e = (pred - t) * 1000
    ev = e[4:11]
    # (b) 12개 전부로 물리 배율 고정 후 치우침 맞춘 잔차 sd (남는 잡음 바닥, 낙관적)
    off12 = (t - s * UM / 1000).mean(); r = (off12 + s * UM / 1000 - t) * 1000
    # 재장착 놓임 사이 sd
    rm = mt[mt.no >= 7].groupby('no').xc.std().mean() * UM
    print(f'{name:10s} 상관 {np.corrcoef(s, t)[0,1]:+.2f} | 평가7 MAE {ev.abs().mean():5.1f} 최대 {ev.abs().max():5.1f} | '
          f'NG1 {pred[11]:.4f} NG2 {pred[12]:.4f} | 12개 잔차 sd {r[r.index<=10].std():.1f}µm(정상10) | 놓임 사이 sd {rm:.1f}µm')
    if name == '같은 px 보정':
        print('   시편별 오차(µm):', ' '.join(f'NO{i}:{e[i]:+.0f}' for i in range(1, 13)))
