# 보고서 8쪽 카드용 오차 예산 그림(세로형, 제목은 카드 머리말이 대신함). 숫자는 fig_budget.py 와 같다
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
B, LB, INK, MUTED, GRID, RED = '#2a78d6', '#9ec5f4', '#2b2b2a', '#7a7a75', '#e6e5e0', '#d03b3b'
rows = [('촬영 중 경계 흐림 변화', 9.4, '18분 동안 -33µm', 'RAW·초점 고정, 기준물 함께 측정', False),
        ('시편별 치우침 (시간 보정 후)', 6.6, '기준값 잡음 포함', '촬영 순서 무작위, 교정 시편 반복', False),
        ('기준값(마이크로미터) 잡음', 5.2, '상한 추정, 위와 겹침', '교정 성적서 있는 마스터', True),
        ('사진 한 장의 경계 잡음', 3.3, '약 0.13px', '이미 0.1px 수준', False),
        ('다시 놓을 때 생기는 몫', 2.7, '', '고정 치구', False)]
fig, ax = plt.subplots(figsize=(6.4, 4.3)); fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.2)
for i, (n, v, d, f, light) in enumerate(rows):
    y = i * 1.25
    ax.text(0, y - 0.42, n, fontsize=11, color=INK, weight='bold', va='center')
    ax.text(11.4, y - 0.42, f, fontsize=9.5, color=MUTED, va='center', ha='right')
    ax.barh(y, v, height=0.42, color=LB if light else B, hatch='//' if light else None, edgecolor='white', lw=0)
    if light: ax.errorbar(v, y, xerr=[[v - 2.8], [7.1 - v]], color=INK, lw=1, capsize=3)
    if v > 8: ax.text(v - 0.2, y, f'{v:.1f}µm  ({d})', fontsize=10, color='white', va='center', ha='right', weight='bold')
    else: ax.text((7.1 if light else v) + 0.2, y, f'{v:.1f}µm' + (f'  ({d})' if d else ''), fontsize=10, color=INK, va='center')
ax.axvline(1.0, color=RED, lw=1.6); ax.text(1.1, -1.05, '요구: 측정 1회 σ 1µm 이하 (%GRR 30%)', fontsize=9.5, color=RED, va='center')
ax.set_xlim(0, 11.4); ax.set_ylim(5.6, -1.3); ax.axis('off')
for xv in [2, 4, 6, 8, 10]: ax.axvline(xv, color=GRID, lw=0.8, zorder=0, ymin=0.05, ymax=0.83)
fig.text(0.02, 0.12, '오른쪽 회색 = 다음 시스템의 대응. 겹치는 기준값을 빼고 합치면 약 12µm(요구의 12배)', fontsize=9, color=MUTED)
fig.text(0.02, 0.05, '원인 아님: 표면 거칠기 · 화소 수     바꿔도 기준선 못 넘음: 대표값 · 실측값 학습 모델', fontsize=9.5, color=INK)
fig.savefig('fig_budget_report.png', dpi=170); print('ok')
