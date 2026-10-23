# 10-21 3차 촬영 분석 코드 (10-21 밤 \~ 10-23)

[3차 촬영 분석 정리](../../docs/05_meetings/2026-10-23_3차촬영_분석정리.md)의 숫자를 만든 코드와 결과 파일이다. 참고용으로 올린다.

- 사진 원본(2억 화소 190장)과 V3 오버레이 그림은 빠져 있다. csv·npz를 읽는 스크립트(`analyze.py`, `grr.py`, `drift_removed.py`, `axial_ana.py`, `noblack_ana.py`, `burst.py`, `pp_check.py`)는 바로 다시 돌릴 수 있다
- 사진 경로는 `../../field/photos/20261021_원전기계/3차_261021`(원본)과 `3차_261021_분류`(사용 190장을 단계·시편 폴더로 나눈 것) 기준으로 적혀 있다(작업 폴더 기준)
- `run_v3_d20.py`는 선행 V3 엔진(`measurement_engine_EDGE_V3_REFLECTION.py`)을 `../../build/demo/hy_system/measurement`에서 불러온다
- `fig_mentor.py`, `fig_mentor_v2.py`는 그림을 `../../ETC/261023/`에 저장한다. 문서에 쓴 그림은 `docs/images/20261021_3rd/`에 복사해 두었다

## 방향 주의
- 사진은 EXIF 방향 3(180도)이다. 코드는 원본 픽셀 배열을 그대로 쓰므로 코드 안의 위·아래(`pin_top`, `sh_top`, "창 맨 위" 등)는 실제와 반대다. 문서와 `fig_photos.py` 그림은 실제 방향으로 적었다

## 50% 경계 두 가지 (중요)
- **처음 방식**(`run_v3_d20.py`의 `tb`, `pin_motion.py`): 측정 창 끝 10% 평균을 바깥 밝기로 쓴다. 암막 사진은 창 맨 아래(원본 좌표로는 맨 위)가 어두운 띠라서 아래 경계를 바닥 쪽에서 잡는 결함이 있다. 결과는 `v3_d20.csv`의 `p50_px`·`pin_px`, `pin_motion.csv`, `ana_p50.txt`, `grr.txt`, `drift_removed_old.txt`, `pp_check.txt`, `burst.txt`
- **국소 50%**(`local_edges.py`): 경계 바로 바깥 20 \~ 60px와 안쪽 10 \~ 30px의 밝기 중간값. 문서의 50% 숫자는 이쪽이다. 결과는 `local_edges.csv`, 그리고 이 값을 `p50_px`·`pin_px` 자리에 넣은 `v3_d20_local.csv`
  - `SRC=v3_d20_local.csv python analyze.py p50_px`처럼 환경변수로 입력을 바꿔 `analyze.py`·`grr.py`·`drift_removed.py`를 돌렸다 -> `ana_local.txt`, `grr_local.txt`, `drift_removed_local.txt`
- V3 엔진 값(`v3_px`)은 이 문제와 무관하다(사전 규칙 판정)

## 순서와 문서 대응
| 파일 | 질문 | 분석 정리 |
|---|---|---|
| `exif_scan.py`, `pos_scan.py`, `classify.py` | 197장을 시간 묶음 + 위치 지문으로 시편별 분류 -> `exif.tsv`, `pos.txt`, `classify.tsv` | 1절 |
| `run_v3_d20.py` | V3 엔진 + 처음 50% 방식으로 Ø20 단 측정(고정 상자 x 8015 \~ 8335) -> `v3_d20.csv` | 2절 |
| `analyze.py` | 사전 규칙 판정, 환산 방식 비교, 재장착, 핀 시간 흐름 -> `ana_v3.txt`(규칙), `ana_p50.txt`, `ana_local.txt` | 2·3절 |
| `pin_motion.py`, `burst.py`, `pp_check.py`, `drift_check.py`, `vblock_check.py` | 핀이 건드려졌나, 연속 촬영 중 변화, 배율·빛 변화인가 | 3절 |
| `local_edges.py` | 국소 50% 방식으로 190장 핀·샤프트 다시 측정 -> `local_edges.csv` | 3·9절 |
| `grr.py` | Gage R&R, MSA 다섯 항목 | 4절 |
| `drift_removed.py` | 시간 변화를 뺐을 때 남는 오차(사후) | 5절 |
| `axial_profile_local.py`, `axial_ana.py` | Ø20 단 안 측정 위치에 따른 지름 -> `axial_profile_local.npz`, `axial_ana.txt`, `axial_profile.png` | 6절 |
| `noblack_ana.py` | 암막 없음 40장 시간 변화·반복성, 두 경계 방식 비교 | 3·7절 |
| `fig_mentor.py`, `fig_mentor_v2.py` | 멘토링 그림(v2 = 국소 50%) | 3·5절 |
| `fig_photos.py` | 실제 사진 그림 5장(사진 전체, Ø20 단 확대, 창 맨 아래 어두운 띠, 핀 경계 단면 앞/뒤, 암막 vs 암막 없음). 원본 사진이 있어야 돌아간다 | 1·3·7·9절 |

## 고쳤던 실수
- 처음 V3 상자를 끝단(약 Ø16.7, x 8900 \~ 9900)에 잡았다. 핀/구간 비율이 맞지 않아 발견하고 Ø20 단으로 바로잡았다
- `local_edges.py` 첫 판은 배경의 1 \~ 3px 어두운 점을 경계로 잡는 버그가 있었다(10px 미만 구간을 빼도록 수정)
