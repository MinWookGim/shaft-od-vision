# 분석 정리 문서에 넣을 실제 사진 그림 (10-23)
# 사진 기반 설명(측정 위치, 창 맨 위 어두운 띠, 핀 경계 모양 변화, 암막 없음 반사띠)마다 원본 사진을 같이 보이게
# 원본 픽셀 배열은 EXIF 방향 3(180도 뒤집힘)이라 축을 둘 다 뒤집어 실제 방향(핀이 샤프트 오른쪽)으로 보인다
# 사진 칸은 축 숫자를 숨기고, 밝기 단면은 경계로부터의 거리(실제 아래쪽이 +)로 그린다
# (10-23 정정: 처음 판은 뒤집힌 채로 보여 위·아래·좌우가 실제와 반대였다)
# 출력: ../../ETC/261023/photo1~5_*.jpg
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
exec(open('local_edges.py').read().split('ps = sorted')[0])     # refine, height, Y0, Y1, PIN_X, SH_X, SHARE
from pos_scan_lib import cross
OUT = Path('../../ETC/261023')
F_EARLY = SHARE / '1_본시험_암막/NO02/20261021_153554.jpg'      # 본시험 앞쪽(NO2 첫 장)
F_MAIN7 = sorted((SHARE / '1_본시험_암막/NO07').glob('*.jpg'))[0]
F_NB7 = sorted((SHARE / '3_암막없음/NO07').glob('*.jpg'))[0]
gray = lambda p: np.asarray(Image.open(p).convert('L'), dtype=float)
flip = lambda a: (a.invert_xaxis(), a.invert_yaxis(), a.set_xticks([]), a.set_yticks([]))   # 180도 돌려 실제 방향으로, 축 숫자 숨김
save = lambda fig, n: (fig.savefig(OUT / n, dpi=110, bbox_inches='tight', pil_kwargs={'quality': 85}), plt.close(fig))

# 1. 사진 전체와 측정 구간
fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
for a, p, t in [(ax[0], F_EARLY, '암막 본시험 (NO2)'), (ax[1], F_NB7, '암막 없음 (NO7)')]:
    a.imshow(np.asarray(Image.open(p).reduce(8).convert('L')), cmap='gray', extent=(0, 16320, 7532, 0))
    a.add_patch(Rectangle((5300, Y0), 500, Y1 - Y0, fill=False, ec='#00bcd4', lw=1.5))
    a.add_patch(Rectangle((8015, Y0), 320, Y1 - Y0, fill=False, ec='#ff5722', lw=1.5))
    a.text(5300, Y1 + 250, '기준 핀 Ø16', color='#00bcd4', fontsize=9, va='bottom', ha='right', bbox=dict(fc='k', alpha=.6, lw=0))
    a.text(8335, Y0 - 250, '측정한 Ø20 단', color='#ff5722', fontsize=9, va='top', bbox=dict(fc='k', alpha=.6, lw=0))
    flip(a); a.set_xlim(13500, 3000); a.set_ylim(1200, 5800); a.set_title(t, fontsize=10)
save(fig, 'photo1_scene.jpg')

# 2. Ø20 단 확대와 국소 50% 경계
g = gray(F_EARLY)
xs = np.arange(7900, 8451, 2)
r = np.array([height(g, x, 0) for x in xs])
fig, ax = plt.subplots(figsize=(7, 8))
ax.imshow(g[3250:4220, 7850:8500], cmap='gray', extent=(7850, 8500, 4220, 3250), vmin=0, vmax=255)
ax.plot(xs, r[:, 1] + Y0, '.', color='#4caf50', ms=2); ax.plot(xs, r[:, 2] + Y0, '.', color='#4caf50', ms=2)
for x in (8015, 8335): ax.axvline(x, color='#ff5722', lw=1.2, ls='--')
flip(ax)
ax.text(7870, 4200, '홈 쪽', color='#ffeb3b', fontsize=11, ha='right', va='top'); ax.text(8490, 4200, '칼라 쪽', color='#ffeb3b', fontsize=11, va='top')
ax.text(8325, 3730, '측정 구간\n(열 32개 중앙값)', color='#ff5722', fontsize=10)
ax.set_title('Ø20 단 확대 (NO2 첫 장). 초록 점 = 열마다 잡은 국소 50% 경계', fontsize=10)
save(fig, 'photo2_d20_edges.jpg')

# 3. 창 맨 위 어두운 띠 -> 처음 방식이 위 경계를 바닥에서 잡은 이유
OY0, OY1 = 3000, 4600                                           # 처음 방식(run_v3_d20.py)의 세로 창
col = np.median(g[OY0:OY1, 8015:8336], axis=1)                  # Ø20 구간 열들의 중앙 밝기
n = len(col) // 2
h_old = (col[:n][:n // 10].mean() + col[:n][-n // 10:].mean()) / 2
t_old = OY0 + np.nanmedian([cross(g[OY0:OY0 + n, x]) for x in SH_X])
t_loc = np.nanmedian(r[(xs >= 8015) & (xs <= 8335), 1]) + Y0
fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
ax[0].imshow(g[OY0:OY1, 7900:8450], cmap='gray', extent=(7900, 8450, OY1, OY0), vmin=0, vmax=255)
ax[0].axhspan(OY0, OY0 + n // 10, color='#e91e63', alpha=.25)
flip(ax[0])
ax[0].text(8440, OY0 + 20, '창 맨 아래 10%\n(어두운 띠)', color='#e91e63', fontsize=9, va='bottom')
ax[0].set_title('처음 방식의 측정 창', fontsize=10)
dd = t_loc - np.arange(OY0, OY1)                               # 국소 50% 경계로부터 거리, 실제 아래쪽이 +
ax[1].plot(dd, col, color='k', lw=1)
ax[1].axhline(h_old, color='#e91e63', ls='--', lw=1); ax[1].text(-140, h_old + 5, f'처음 방식의 50% 문턱 {h_old:.0f}', color='#e91e63')
ax[1].axvline(t_loc - t_old, color='#e91e63', lw=1.2); ax[1].axvline(0, color='#4caf50', lw=1.2)
ax[1].text(t_loc - t_old - 4, 200, f'처음 방식 아래 경계\n({t_old - t_loc:.0f}px 안쪽)', color='#e91e63', fontsize=9, ha='right')
ax[1].text(4, 120, '국소 50%', color='#4caf50', fontsize=9)
ax[1].set_xlim(-150, t_loc - OY0); ax[1].set_xlabel('경계로부터 거리 (px)   위(샤프트 안) <-  -> 아래(배경, 창 끝)'); ax[1].set_ylabel('밝기 (Ø20 구간 열 중앙값)')
ax[1].set_title('샤프트 아래 경계 밝기 단면: 창 끝이 어두운 띠라서 문턱이 바닥 근처로 내려간다', fontsize=10)
ax[1].grid(alpha=.3)
save(fig, 'photo3_window_band.jpg')

# 4. 핀 경계 밝기 모양: 본시험 앞(NO2 10장) vs 뒤(NO11 10장). 한 장끼리는 잡음이 커서 10장 평균으로
ge = g
row = [d for d in csv.DictReader(open('local_edges.csv')) if d['stage'] == '1']
pe = {k: np.mean([float(d[k]) for d in row if d['no'] == 'NO02']) for k in ('pin_top', 'pin_bot')}
pl = {k: np.mean([float(d[k]) for d in row if d['no'] == 'NO11']) for k in ('pin_top', 'pin_bot')}
def mean_prof(no, a, b):
    return np.mean([np.median(gray(q)[a:b, 5300:5800], axis=1) for q in sorted((SHARE / '1_본시험_암막' / no).glob('*.jpg'))], axis=0)
P = {(no, lo): mean_prof(no, lo, hi) for no in ('NO02', 'NO11') for lo, hi in ((3995, 4055), (3385, 3440))}
prof = lambda G, a, b: P[(G, a)]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={'width_ratios': [1.1, 1, 1]})
ax[0].imshow(ge[3330:4110, 5250:5850], cmap='gray', extent=(5250, 5850, 4110, 3330), vmin=0, vmax=255)
ax[0].axhspan(3385, 3440, color='#2196f3', alpha=.25); ax[0].axhspan(3995, 4055, color='#ff9800', alpha=.25)
flip(ax[0]); ax[0].set_title('핀 Ø16 (NO2 첫 장), 경계 두 곳', fontsize=10)
for a, (lo, hi), k, c, nm in [(ax[1], (3995, 4055), 'pin_bot', '#ff9800', '위 경계'), (ax[2], (3385, 3440), 'pin_top', '#2196f3', '아래 경계')]:
    e0 = float(pe[k]); d = e0 - np.arange(lo, hi)                 # 앞 사진 경계로부터 거리, 실제 아래쪽이 +
    a.plot(d, prof('NO02', lo, hi), color='k', lw=1.4, label='본시험 앞 NO2 (10장 평균)')
    a.plot(d, prof('NO11', lo, hi), color='#d32f2f', lw=1.4, ls='--', label=f'본시험 뒤 NO11 (10장 평균, {abs(float(pl[k]) - e0):.2f}px 이동)')
    a.axvline(0, color='k', lw=.8, ls=':'); a.axvline(e0 - float(pl[k]), color='#d32f2f', lw=.8, ls=':')
    a.set_title(f'{nm} 밝기 단면', fontsize=10); a.set_xlabel('경계로부터 거리 (px, 아래쪽이 +)'); a.legend(fontsize=8); a.grid(alpha=.3)
ax[1].set_ylabel('밝기')
fig.suptitle('같은 핀, 약 17분 차이: 위 경계 단면 모양이 바뀌고, 아래 경계는 거의 그대로다 (점선 = 국소 50% 경계)', fontsize=11)
save(fig, 'photo4_pin_edge_profile.jpg')

# 5. 암막 vs 암막 없음 (같은 NO7)
fig, ax = plt.subplots(2, 1, figsize=(12, 6.8)); fig.subplots_adjust(hspace=.3)
for a, p, t in [(ax[0], F_MAIN7, '암막 본시험 (NO7): 뒤에서 오는 빛만, 물체는 검은 그림자'),
                (ax[1], F_NB7, '암막 없음 (NO7): 앞빛이 들어와 핀·샤프트 몸통에 하얀 반사띠가 생긴다')]:
    a.imshow(np.asarray(Image.open(p).crop((5000, 3150, 8700, 4350)).reduce(2).convert('L')), cmap='gray',
             extent=(5000, 8700, 4350, 3150), vmin=0, vmax=255)
    flip(a); a.set_title(t, fontsize=10); a.set_yticks([])
save(fig, 'photo5_blackout_vs_none.jpg')
print('t_old', t_old, 't_loc', t_loc, 'h_old', h_old)
