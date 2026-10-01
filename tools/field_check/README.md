# field_check — 10-02 현장 즉석 확인

체크리스트 A1·A3·A4·A5와 멈출 신호용. 사진 1장 2초 안팎(200MP).

```
# 1) A3 사진 1장으로 좌표 눈금 그림 만들기
conda run -n aiffel python field_check.py grid 사진.jpg      -> 사진_grid.jpg

# 2) site_template.json 을 site.json 으로 복사해 roi 4개 숫자 채우기

# 3) 확인
conda run -n aiffel python field_check.py check site.json 사진.jpg [사진2.jpg ...]
   -> 표(샤프트 px, 핀 px, 각 sd, 배경 포화 %, 대비) + 사진_check.jpg(경계 점 찍은 확대 크롭)
```

| 단계 | 넣을 사진 | 볼 것 |
|---|---|---|
| A1 | 백라이트만 | 포화 % (1% 넘으면 경고) |
| A3 | 차광 후 1장 | 사진_check.jpg 에서 점이 실루엣 가장자리에 붙었나, sd 경고 없나, 샤프트 px 400 이상 |
| A4 | 연속 3장 한꺼번에 | [A4] 표 — 첫 장 대비 0.1% 이내 |
| A5 | "중" 시편 | site.json 에 `true_shaft_mm` 실측값 -> [A5] 환산·핀 이동 안내 |
| 블록 B~ | 시편마다 1장 | site.json 에 `ref_pin_px`(A4 기준값) 적어 두면 매번 기준 대비 % |

- 핀 방식 2(눕힘)면 `"axis": "v"`
- 경고 `경계를 찾은 줄이 …뿐` -> ROI 에 받침·핀 끝(모따기)이 섞임. 고른 구간으로 좁힌다
- 경고 `줄마다 폭 흔들림` -> 반사 띠·테이퍼·먼지. 10-01 집 사진(140524)에서 세운 핀 옆면 회색 반사 띠로 9% 흔들림이 나왔다

검증: `test_field_check.py` (합성 사진으로 소수점 폭 ±0.05px, 0.1% 변화 검출, ROI 끝 어두운 물체, A5 환산·이동 방향 — 전부 통과)
