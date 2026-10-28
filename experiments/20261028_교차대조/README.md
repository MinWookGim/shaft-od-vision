# 10-28 팀장·팀원 분석 교차 대조

- 사진은 저장소에 없다(팀장 로컬). 코드는 참고용이며 경로는 팀장 작업 폴더 기준. 팀원 측정값은 [gon311/final_project](https://github.com/gon311/final_project) `experiments/3rd/E6_E10_20261021/results/values_by_image_1021.csv`를 인자로 넘긴다
- `cross_calib.py <팀원 csv>`: 경계 원천 4가지 x 교정 방식 7가지, 정상 9개 하나씩 빼고 맞히기 -> `cross_calib.csv` (①)
- `fine_cols.py`: 시편마다 첫 사진 Ø20 단 1px 열 경계 -> 시편 공통으로 튀는 열 찾기 (②, 출력 npy는 저장소에 없음)
- `repr_clean.py`: 대표값 후보 8가지 x 31열/21열 x 교정 4가지 -> `repr_clean.csv` (②)
- `remount_cross.py <팀원 csv>`: 재장착 반복성 정의 2가지 x 경계 4가지 -> `remount_cross.csv` (③)
- `texture.py`, `texture_aligned.py`: 샤프트 아래 밝은 띠(역광판) 질감 이동·상관 -> `texture*.csv`, `timeline.csv` (④)
- `edge_width.py`: 핀·샤프트 경계 10~90% 폭과 밝기 -> `edge_width.csv` (④)
- `fig_timeline.py`: 그림 -> `docs/images/20261028_cross/fig_timeline.png`
- 정리: `docs/05_meetings/2026-10-28_교차대조.md`
- `fig_budget.py`, `fig_budget_report.py`: 오차 예산 그림(문서용 / 보고서 8장 카드용)
- `notion_figs.py`: 팀 노션 공유용 그림 5장 -> `docs/images/20261028_cross/fig1 ~ fig5`
