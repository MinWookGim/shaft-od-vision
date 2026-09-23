# 2026-09-23 — 질문: 원본 해상도에서 31점 엣지 검출이 왜 흔들리나, YOLO 박스로 제한하면 되나

- `check.py` A2(시편1 2번째, `20260916_154320.jpg`) 원본 6008px vs 2992px 축소본에서 `measure_roi`의 std_pixel(31점 산포) 대조
- `visualize.py` 위 두 조건의 31개 측정점을 실제 사진 위에 찍어서 비교 → `docs/images/20260923_edge_report/A2_31점_비교.png`
- `test_box_restrict.py` 원본 6008px에서 `vertical_margin`·`dia_basis` 조합 5가지 대조(A2 한 장)
- `test_all14.py` 위에서 제일 나은 조합(`vertical_margin=0.05, dia_basis="box"`)을 시편1 14장 전체에 적용, 기본값과 대조
- `line_fit_combo.py` 다겸님 제안(직선에서 벗어난 점 반복 제거 후 재적합)으로 라인 피팅을 강화해서, 위 조합과 같이 썼을 때 A2 한 장·시편1 14장에서 어떻게 되는지 대조
- `count_wrong.py` 31개 점을 정답 위치와 하나씩 대조해 몇 개가 어디를 잡았는지 센다(14 대 17)
- `noise_floor.py` 장간 sd 두 개의 차이가 표본 노이즈보다 큰지 부트스트랩으로 확인
- `make_charts.py` 문서용 차트 5장을 `docs/images/20260923_edge_report/` 에 생성
- `paths.py` 경로를 한 곳에서 정한다. 엔진·현장사진은 저장소 밖이라 환경변수(`SHAFT_ENGINE_DIR`/`SHAFT_FIELD_DIR`) 또는 `paths_local.py`(git 제외)로 각자 지정한다
- 결과: `../results/20260923/` · 해석·팀 공유용 정리: `docs/05_meetings/2026-09-23_엣지검출_원본해상도_실험.md`
