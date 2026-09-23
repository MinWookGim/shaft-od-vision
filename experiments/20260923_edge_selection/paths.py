"""이 폴더 실험 스크립트들이 쓰는 경로를 한 곳에서 정한다.

측정 엔진(`shaft_core.py`)과 현장 사진·모델은 **이 저장소 밖**에 있고, 사람마다 둔 위치가 다르다.
그래서 절대경로를 코드에 박지 않고 아래 순서로 찾는다.

  1. 환경변수 `SHAFT_ENGINE_DIR` / `SHAFT_FIELD_DIR`
  2. 이 폴더의 `paths_local.py` (git 에 안 올라간다. 각자 자기 경로를 적어두면 된다)
  3. 둘 다 없으면 무엇을 설정해야 하는지 알려주고 멈춘다

`paths_local.py` 는 이렇게 두 줄이면 된다:

    ENGINE_DIR = "/내/컴퓨터의/샤프트측정/experiments"   # shaft_core.py 가 있는 폴더
    FIELD_DIR  = "/내/컴퓨터의/Pre365/field"             # photos/ 와 models/ 를 품은 폴더

저장소 안에 있는 경로(결과·이미지 출력 위치)는 이 파일 위치를 기준으로 자동 계산하므로 설정이 필요 없다.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                                   # shaft-od-vision/

# 저장소 안 경로 — 설정 없이 항상 쓸 수 있다
IMG_OUT = REPO / "docs" / "images" / "20260923_edge_report"
RESULT_OUT = REPO / "experiments" / "results" / "20260923"

# 이번 실험의 대표 사진 (시편1 14회 촬영 중 2번째 = 문서의 A2)
A2_NAME = "20260916_154320.jpg"
PHOTO_SUBDIR = Path("photos") / "20260916_원전기계"
MODEL_SUBPATH = Path("models") / "demo_yolo11n_20260917.pt"

try:
    import paths_local as _local
except ModuleNotFoundError:
    _local = None


def _resolve(attr, env_key, what):
    """환경변수 -> paths_local.py 순으로 찾는다. 못 찾으면 무엇을 설정해야 하는지 알려주고 멈춘다."""
    value = os.environ.get(env_key) or getattr(_local, attr, None)
    if value is None:
        raise SystemExit(
            f"[경로 설정 필요] {what}을(를) 못 찾았다.\n"
            f"  방법 1: 환경변수 {env_key}=/그/경로 를 지정한다\n"
            f"  방법 2: {HERE / 'paths_local.py'} 에 {attr} = \"/그/경로\" 한 줄을 적는다\n"
            f"  (자세한 설명은 같은 폴더 paths.py 맨 위 주석)"
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"[경로 없음] {attr} 로 지정한 {path} 가 실제로 없다.")
    return path


def engine_dir():
    """measurement 엔진(shaft_core.py)이 있는 폴더."""
    return _resolve("ENGINE_DIR", "SHAFT_ENGINE_DIR", "측정 엔진 폴더(shaft_core.py 가 있는 곳)")


def field_dir():
    """09/16 현장 촬영분(photos/, models/)을 품은 폴더."""
    return _resolve("FIELD_DIR", "SHAFT_FIELD_DIR", "현장 데이터 폴더(photos/ 와 models/ 가 있는 곳)")


def photo_dir():
    """09/16 현장 사진이 들어 있는 폴더."""
    return field_dir() / PHOTO_SUBDIR


def photo(name=A2_NAME):
    """사진 한 장의 경로. 이름을 안 주면 이번 실험의 대표 사진(A2)."""
    return photo_dir() / name


def model_path():
    return field_dir() / MODEL_SUBPATH


def load_core():
    """shaft_core 를 import 하고 모델 경로까지 맞춰서 돌려준다.

    09/16 구도에서는 재학습 모델을 쓴다 (7월 best.pt 는 이 구도에서 검출 자체가 약하다).
    """
    sys.path.insert(0, str(engine_dir()))
    import shaft_core as core
    core.MODEL_PATH = str(model_path())
    return core
