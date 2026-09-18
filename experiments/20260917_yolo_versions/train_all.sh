#!/usr/bin/env bash
# 같은 데이터·같은 설정으로 아키텍처만 바꿔 학습. 결과는 runs/<이름>/
cd "$(dirname "$0")"; source ~/anaconda3/etc/profile.d/conda.sh; conda activate aiffel
for M in yolov8n yolov10n yolo11n yolo12n yolo26n yolo26s yolo11s; do
  echo "=== $M $(date +%H:%M:%S)"
  python -c "
from ultralytics import YOLO
m = YOLO('${M}.pt')
m.train(data='ds/data.yaml', epochs=60, imgsz=640, batch=8, seed=0, deterministic=True, patience=60,
        project='runs', name='${M}', exist_ok=True, verbose=False, plots=False, workers=4)
" 2>&1 | grep -E "^ *(60/60|Epoch|all)|Error|error" | tail -4
done
echo "=== done $(date +%H:%M:%S)"; touch done.flag
