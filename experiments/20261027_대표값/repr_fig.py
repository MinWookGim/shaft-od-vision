# 대표값 후보별 오차(하나씩 빼고 맞히기, 시간 보정 후)와 사진 사이 흔들림. 두 설정(본시험만 / 3차 분석정리 설정)
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'); plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
a = pd.read_csv('repr_summary_main.csv'); b = pd.read_csv('repr_summary_3rd.csv')
a = a[a.시간보정 == '후'].set_index('대표값'); b = b[b.시간보정 == '후'].set_index('대표값')
names = list(a.index); y = range(len(names))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
ax[0].barh([i + 0.2 for i in y], a.B_MAE, height=0.38, color='#2f6fd0', label='본시험 10장, 핀 기준 배율')
ax[0].barh([i - 0.2 for i in y], b.B_MAE, height=0.38, color='#9db8e8', label='3차 분석정리 설정(재장착 포함)')
ax[0].axvline(6.9, color='#d9463e', ls='--', lw=1.5, label='기준선 6.9µm (측정 없이 평균 부르기)')
ax[0].set_yticks(list(y)); ax[0].set_yticklabels(names); ax[0].invert_yaxis()
ax[0].set_xlabel('정상 10개 하나씩 빼고 맞히기 MAE (µm, 작을수록 좋음)'); ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(axis='x', alpha=.3)
ax[0].set_title('실측과의 오차 (시간 보정 후)', fontsize=11); ax[0].set_xlim(0, 62)
ax[1].barh(list(y), a.사진사이sd, color='#8a91a0'); ax[1].set_xlabel('같은 놓임 사진 사이 sd (µm)'); ax[1].grid(axis='x', alpha=.3)
ax[1].set_title('사진마다 흔들리는 정도', fontsize=11)
fig.suptitle('대표값 후보 비교 (3차 사진·실측): 가운데 값(평균·중앙값)이 낫고, 끝값(Max·envelope)은 잡음을 따라간다', fontsize=11.5)
fig.tight_layout(); fig.savefig('fig_repr.png', dpi=120)
