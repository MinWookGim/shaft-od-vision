# 핀 위치(s)별 위·아래 경계와 폭: 130장 평균 (각 사진의 s 20~120 평균을 0 으로 맞춤)
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumSquare' in f and 'Bold' not in f and 'Extra' not in f: fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['NanumSquare', 'DejaVu Sans']
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, '_out', 'pin_scan.npz')); XS, TOP, BOT = z['XS'], z['TOP'], z['BOT']; W = BOT - TOP
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
acc = {'위 경계(아래로 +)': {}, '아래 경계': {}, '폭': {}}
for k in range(len(W)):
    ok = np.where((W[k] > 250) & (XS >= 9500))[0]; i0 = ok[0]; i1 = i0
    while i1 + 1 < len(XS) and not np.isnan(W[k, i1 + 1]) and W[k, i1 + 1] > 250: i1 += 1
    s = XS[i0:i1 + 1] - XS[i0]; m = (s >= 20) & (s <= 120)
    for name, arr in (('위 경계(아래로 +)', TOP[k, i0:i1 + 1]), ('아래 경계', BOT[k, i0:i1 + 1]), ('폭', W[k, i0:i1 + 1])):
        a = arr - np.nanmean(arr[m])
        for ss, v in zip(s, a): acc[name].setdefault(ss, []).append(v)
for name, c in zip(acc, ('#d62728', '#1f77b4', 'k')):
    ss = np.array(sorted(k for k in acc[name] if len(acc[name][k]) > 60)); mu = np.array([np.mean(acc[name][k]) for k in ss])
    ax[0].plot(ss, mu, color=c, label=name)
ax[0].axhline(0, color='gray', lw=.5); ax[0].set_xlim(15, 180); ax[0].set_ylim(-3, 3)
ax[0].set_xlabel('핀 왼쪽 끝에서 거리 s (px, 오른쪽 = V블록 끝 쪽)'); ax[0].set_ylabel('px (s 20~120 평균 = 0)'); ax[0].legend()
ax[0].set_title('130장 평균: 폭 변화는 위 경계에서 온다')
r = os.path.join(HERE, '_out', 'analyze.txt'); L = open(r).read().split('\n')
g = [l for l in L if l.startswith('묶음')]
xl = [float(l.split('xL ')[1].split()[0]) for l in g]; pa = [float(l.split('핀A ')[1].split()[0]) for l in g]; pb = [float(l.split(' B ')[1].split()[0]) for l in g]
foc = [l.split()[1] if l[2] != ' ' else l.split()[2] for l in g]
idx = np.arange(1, 14)
ax[1].plot(idx, pa, 'o-', label='A 고정 열 9655~9700'); ax[1].plot(idx, pb, 's-', label='B 핀 위 같은 자리 s 60~105')
ax[1].axvspan(.5, 4.5, color='#eee'); ax[1].axvspan(6.5, 13.5, color='#e8f0ff')
ax[1].text(1, 271.1, 'F1', fontsize=9); ax[1].text(5, 271.1, 'F2', fontsize=9); ax[1].text(10, 271.1, 'F3', fontsize=9)
ax[1].set_xticks(idx); ax[1].set_xlabel('묶음 (1·2 = 핀이 약 40px 왼쪽에 놓임)'); ax[1].set_ylabel('핀 폭 평균 (px)'); ax[1].legend(fontsize=8)
ax[1].set_title('묶음별 핀 폭: 같은 자리를 재면 F1 안 흔들림이 1.11 -> 0.71px')
fig.tight_layout(); fig.savefig(os.path.join(HERE, '_out', 'pin_profile.png'), dpi=110)
