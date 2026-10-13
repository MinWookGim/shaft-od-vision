# 실험 2 학습: yolo11n(가장 작은 판, 약 2.6M 파라미터)을 10-08 사진으로. 2겹(A 로 학습 -> B 로 검증, 반대도)
# 카메라가 움직일 때를 대비해 위치·크기 흔들기를 크게 준다(translate 0.3, scale 0.5)
import os, sys
from ultralytics import YOLO
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
W0 = os.path.join(ROOT, '..', '..', 'build', 'yolo_versions', 'yolo11n.pt')
for fold in sys.argv[1:] or ['A', 'B']:
    m = YOLO(W0)
    m.train(data=os.path.join(ROOT, '_data', 'pos', f'train_{fold}.yaml'), imgsz=1280, epochs=80, batch=8, device=0,
            translate=0.3, scale=0.5, fliplr=0.5, mosaic=1.0, seed=0, workers=4, plots=False,
            project=os.path.join(ROOT, 'runs'), name=f'pos_{fold}', exist_ok=True, verbose=False)
