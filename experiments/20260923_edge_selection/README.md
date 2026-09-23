# 2026-09-23 — 질문: 원본 해상도에서 31점 엣지 검출이 왜 흔들리나, YOLO 박스로 제한하면 되나

- `check.py` A2(시편1 2번째, `20260916_154320.jpg`) 원본 6008px vs 2992px 축소본에서 `measure_roi`의 std_pixel(31점 산포) 대조
- `visualize.py` 위 두 조건의 31개 측정점을 실제 사진 위에 찍어서 비교 → `../results/20260923/A2_31점_비교.png`
- `test_box_restrict.py` 원본 6008px에서 `vertical_margin`·`dia_basis` 조합 5가지 대조(A2 한 장)
- `test_all14.py` 위에서 제일 나은 조합(`vertical_margin=0.05, dia_basis="box"`)을 시편1 14장 전체에 적용, 기본값과 대조
- `line_fit_combo.py` 다겸님 제안(직선에서 벗어난 점 반복 제거 후 재적합)으로 라인 피팅을 강화해서, 위 조합과 같이 썼을 때 A2 한 장·시편1 14장에서 어떻게 되는지 대조
- 결과: `../results/20260923/` · 해석·팀 공유용 정리: `docs/05_meetings/2026-09-23_엣지검출_원본해상도_실험.md`
