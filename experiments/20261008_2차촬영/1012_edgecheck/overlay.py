# 멘토 조언 1 (10-10): "측정 전에 경계를 제대로 잡는지 이미지로 먼저 확인하라"
# 13묶음마다 가운데 사진 한 장(5번째)을 골라 샤프트 위·아래 경계, 핀 위·아래 경계를 크게 잘라
# 우리 방법(M0 50% 경계, 빨강)과 바깥 단 방법(M2, 하늘색)이 잡은 선을 그 위에 그린다
# 사진은 1008_methods/_out/cache.npz (경계 근처만 잘라둔 것) 를 쓴다 -> 2억 화소 원본을 다시 안 읽음
import os, sys, csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '1008_methods'))
import methods as M                                      # 경계 찾는 함수·캐시를 그대로 재사용

rows = M.rows
byg = {}
for r in rows: byg.setdefault(int(r['묶음']), []).append(r)
W = 40                                                   # 경계 위아래로 보여줄 폭(px)

def edges_of(rec):
    """사진 한 장에서 열마다 M0·M2 경계를 구해 중앙값으로. 샤프트(위,아래), 핀(위,아래)"""
    sh, pin, bg = rec['shaft'].astype(float), rec['pin'].astype(float), rec['bg']
    out = {}
    for name, f in (('M0', M.m0_half), ('M2', M.m2_outer)):
        s = np.array([f(sh[:, x], bg, M.S_LO, M.S_HI) for x in range(0, sh.shape[1], 2)])
        p = np.array([f(pin[:, x], bg, M.P_LO, M.P_HI) for x in range(35, 80, 2)])
        out[name] = (np.nanmedian(s[:, 0]), np.nanmedian(s[:, 1]), np.nanmedian(p[:, 0]), np.nanmedian(p[:, 1]))
    return out

def panel(ax, img, y, bg, lines, title):
    """경계 y 위아래 W px 를 잘라 보여준다. 밝기를 [어두운쪽, 배경] -> [0,1] 로 늘리고 감마 0.5 로 연한 띠를 드러냄"""
    a = img[int(y) - W:int(y) + W].astype(float)
    lo = np.percentile(img, 5)
    v = np.clip((a - lo) / (bg - lo), 0, 1) ** 0.5
    ax.imshow(v, cmap='gray', aspect='auto', interpolation='nearest', extent=(0, a.shape[1], int(y) + W, int(y) - W))
    for yy, c in lines: ax.axhline(yy, color=c, lw=1.2)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.tick_params(labelsize=6)

def profile(ax, img, ys, bg, lines):
    """같은 자리의 열 평균 밝기 곡선(배경=1). 아래 경계 안쪽의 '선반'(연한 띠)이 보이는지"""
    y = int(ys)
    prof = img[y - W:y + W].astype(float).mean(1) / bg
    ax.plot(np.arange(y - W, y + W), prof, 'k', lw=1)
    for yy, c in lines: ax.axvline(yy, color=c, lw=1.2)
    ax.set_ylim(0, 1.05); ax.tick_params(labelsize=6); ax.grid(alpha=.3)

summary = []
fig, axs = plt.subplots(13, 5, figsize=(15, 34))
for i, g in enumerate(sorted(byg)):
    r = byg[g][4]; rec = M.DATA[r['파일']]
    e = edges_of(rec); bg = rec['bg']
    sh, pin = rec['shaft'], rec['pin']
    tag = f"{r['시편']} ({r['초점']}, 묶음{g})"
    L = lambda k: [(e['M0'][k], 'red'), (e['M2'][k], 'deepskyblue')]
    panel(axs[i, 0], sh, e['M0'][0], bg, L(0), f'{tag} 샤프트 위')
    panel(axs[i, 1], sh, e['M0'][1], bg, L(1), f'{tag} 샤프트 아래')
    profile(axs[i, 2], sh, e['M0'][1], bg, L(1)); axs[i, 2].set_title('샤프트 아래 밝기 곡선', fontsize=8)
    panel(axs[i, 3], pin, e['M0'][2], bg, L(2), '핀 위')
    panel(axs[i, 4], pin, e['M0'][3], bg, L(3), '핀 아래')
    summary.append([r['파일'], g, r['시편'], r['초점'], *(f'{v:.1f}' for v in e['M0'] + e['M2'])])
fig.suptitle('10-08 경계 확인 — 빨강 = M0(우리 50% 경계), 하늘색 = M2(가장 바깥 단). 묶음마다 5번째 사진', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(os.path.join(HERE, '_out', 'edge_overlay_13.png'), dpi=110)
csv.writer(open(os.path.join(HERE, '_out', 'edge_overlay_13.csv'), 'w', newline='', encoding='utf-8-sig')).writerows(
    [['파일', '묶음', '시편', '초점', 'M0_샤프트위', 'M0_샤프트아래', 'M0_핀위', 'M0_핀아래', 'M2_샤프트위', 'M2_샤프트아래', 'M2_핀위', 'M2_핀아래']] + summary)
print('saved')

# ---- 보고용 작은 그림: 초점 구간마다 하나씩 (NO4=F1, NO6=F2, NO8=F3). 밝기 곡선에 50% 문턱(점선)도 그린다 ----
fig, axs = plt.subplots(3, 3, figsize=(12, 8))
for i, g in enumerate((4, 6, 8)):
    r = byg[g][4]; rec = M.DATA[r['파일']]; e = edges_of(rec); bg = rec['bg']; sh = rec['shaft']
    L = lambda k: [(e['M0'][k], 'red'), (e['M2'][k], 'deepskyblue')]
    tag = f"{r['시편']} ({r['초점']})"
    panel(axs[i, 0], sh, e['M0'][0], bg, L(0), f'{tag} 샤프트 위 경계')
    panel(axs[i, 1], sh, e['M0'][1], bg, L(1), f'{tag} 샤프트 아래 경계')
    profile(axs[i, 2], sh, e['M0'][1], bg, L(1))
    seg = sh[M.S_LO:M.S_HI].astype(float)
    thr = np.median([(bg + np.percentile(seg[:, x], 5)) / 2 for x in range(0, seg.shape[1], 2)]) / bg
    axs[i, 2].axhline(thr, color='gray', ls='--', lw=1); axs[i, 2].set_title('아래 경계 밝기 곡선 (점선 = 50% 문턱)', fontsize=8)
fig.suptitle('빨강 = 우리 50% 경계, 하늘색 = 가장 바깥 단. 위는 한 칸 절벽, 아래는 연한 띠 + 계단', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(HERE, '_out', 'edge_overlay_3.png'), dpi=120)
print('saved small')
