# 10-27 백라이트 안정성 확인 (3차 본시험·재장착 140장)

- `light_check.py`·`analyze_light.py`: 밝은 배경 R·G·B, R/B, 줄무늬, 핀 px와의 관계
- `mid_check.py`·`analyze_mid.py`: V블록 옆면·바닥 같은 중간 밝기 영역, 테스트5 대비 감도 확인. 그림 `docs/images/20261027_dist/fig_mid_light.png`
- `wb_flag.py`: EXIF 화이트밸런스(자동 139/140)
- 결론: 폰이 밝기·색을 맞춰 JPEG로는 조명 변화를 감지할 수 없었다(판단 불가). 정리는 최종보고서 공유 문서 "조명 안정성" 절
