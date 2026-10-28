# 오차 예산 그림: 측정 1회 오차를 원인별로(µm, 1σ 상당) 놓고 %GRR 30% 요구선(σ 1µm)과 비교
# 공통 틀: 팀원 cross50 경계(가장 잡음 작은 측정), 물리 배율, 정상 9개(NO2~10) — cross_calib.csv, remount_cross.csv, timeline.csv
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
B, LB, INK, MUTED, GRID = '#2a78d6', '#9ec5f4', '#2b2b2a', '#8a8a85', '#e6e5e0'
rows = [  # 이름, 1σ 상당 µm, 값 설명, 다음 시스템 대응, 연한 막대 여부
    ('촬영 중 경계 흐림 변화', 9.4, '18분 동안 -33µm (균일 흐름의 1σ 상당)', 'RAW·초점 고정, 기준물을 같은 방법으로 함께 측정', False),
    ('시편별 치우침 (시간 보정 후)', 6.6, '정상 9개 하나씩 빼기 오차 sd (기준값 잡음 포함)', '촬영 순서 무작위, CAL 처음·중간·끝 반복', False),
    ('기준값(마이크로미터) 잡음', 5.2, '상한 추정 (95% 2.8 ∼ 7.1), 위 치우침과 겹침', '교정 성적서 있는 마스터, 같은 위치 반복 측정', True),
    ('사진 한 장의 경계 잡음', 3.3, '같은 놓임 연속 촬영 sd (약 110열 평균, 0.13px)', '열을 많이 평균 (이미 0.1px 수준)', False),
    ('다시 놓을 때 생기는 몫', 2.7, '재장착 6번 놓임', 'V블록·멈춤 고정 치구', False),
]
fig, ax = plt.subplots(figsize=(11, 5.6)); fig.subplots_adjust(left=0.24, right=0.62, top=0.8, bottom=0.27)
y = np.arange(len(rows))
for i, (n, v, d, f, light) in enumerate(rows):
    ax.barh(i, v, height=0.56, color=LB if light else B, hatch='//' if light else None, edgecolor='white', lw=0)
    if light: ax.errorbar(v, i, xerr=[[v - 2.8], [7.1 - v]], color=INK, lw=1, capsize=3)
    ax.text(v + (2.2 if light else 0.25), i - 0.05, f'{v:.1f}', va='center', fontsize=10.5, color=INK, weight='bold')
    ax.text(0.15, i + 0.42, d, fontsize=8, color=MUTED, va='center')
    ax.text(1.03, i, f, transform=ax.get_yaxis_transform(), fontsize=9, color=INK, va='center', clip_on=False)
ax.text(1.03, -0.75, '다음 시스템에서 잡을 장치', transform=ax.get_yaxis_transform(), fontsize=9.5, color=MUTED, weight='bold', va='bottom', clip_on=False)
ax.axvline(1.0, color='#d03b3b', lw=1.6); ax.text(1.12, -0.75, '요구: 측정 1회 σ 1µm 이하\n(%GRR 30%, 공차 ±10µm)', fontsize=8.5, color='#d03b3b', va='bottom')
ax.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK); ax.invert_yaxis()
ax.set_xlim(0, 11); ax.set_ylim(len(rows) - 0.4, -0.95); ax.set_xlabel('측정 1회 오차에 더해지는 크기 (µm, 1σ 상당)', color=INK)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK, labelsize=9.5)
fig.text(0.02, 0.965, '오차 예산: 무엇이 몇 µm씩 판정을 막았나 (3차 촬영)', fontsize=14, color=INK, weight='bold', va='top')
fig.text(0.02, 0.91, '모두 같은 틀(가장 잡음이 작은 경계, 물리 배율, 정상 시편 9개)에서 측정. 겹치는 기준값을 빼고 합치면 약 12µm로 요구의 약 12배',
         fontsize=9.5, color=MUTED, va='top')
fig.text(0.02, 0.15, '원인·병목이 아니었던 것', fontsize=10, color=INK, weight='bold')
fig.text(0.02, 0.095, '표면 거칠기 (세 분석 모두 원인으로 지지하는 결과 없음)  ·  화소 수 (2배 축소해도 오차 같음)  ·  대표값 선택 (어느 후보도 기준선 못 넘음)  ·  실측값 학습 모델 (8개 변형 모두 개선 없음)',
         fontsize=9, color=MUTED)
fig.text(0.02, 0.035, '흐림 변화는 18분 동안 한 방향으로 흐른 양을 균일 분포의 1σ(폭/√12)로 바꾼 값이다. 기준값 잡음은 오차 RMSE에서 거꾸로 구한 상한이라 위 치우침에 이미 들어 있다.',
         fontsize=8, color=MUTED)
fig.savefig('fig_budget.png', dpi=140); print('ok')
