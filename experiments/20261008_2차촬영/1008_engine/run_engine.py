# 10-08 본촬영 130장을 선행 개발자 측정 엔진(shaft_core.py, 원본 재현 검증 통과)으로 돌린다
# 질문 1: YOLO 가 10-08 사진에서 측정 위치를 찾나
# 질문 2: 엔진이 잡은 위·아래 엣지가 어디인가 (멘토: "경계를 제대로 잡는지 먼저 봐라")
# 질문 3: 엔진 옵션(엣지 고르는 규칙, 축소 배율)을 바꾸면 결과가 어떻게 바뀌나
import sys, os, csv
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('SHAFT_CORE_DIR', '.'))   # 선행 엔진 shaft_core.py 가 있는 폴더
sys.path.insert(0, os.path.join(HERE, '..', '1008_field'))
import shaft_core as sc
from t1_analyze import edges          # 내 방법(50% 밝기 경계), 비교용

D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
OUT = os.path.join(HERE, '_out')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))

# 측정 위치(박스). 선행 개발자도 10-08 은 모델 대신 직접 지정했다고 함
# x = 사진마다 자동으로 찾은 베어링부(칼라 끝~홈 앞), y = 샤프트 위아래 여유를 둔 고정 높이
Y1, Y2 = 3600, 4320            # 샤프트(약 544px)를 품는 고정 높이 -> 엔진이 박스 높이의 40% 를 위아래로 더 넓힌다
PIN = (9620, 3790, 9840, 4120)  # 핀게이지 어두운 부분 (x1, y1, x2, y2)

VARIANTS = {   # 이름: (축소배율, 엔진 옵션)
    'A_원본규칙_원본해상도': (1.0, sc.Params()),
    'B_원본규칙_절반축소':   (0.5, sc.Params()),
    'C_가장센엣지_원본해상도': (1.0, sc.Params(edge_select='strongest')),
    'D_중앙값_원본해상도':   (1.0, sc.Params(edge_select='median')),
}

def run_box(img, box, p, scale, keep_edges=False):
    """measure_roi 와 같은 순서지만, 열마다 잡은 위·아래 엣지 위치도 돌려준다(그림용)."""
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r = sc.measure_roi(img, (x1, y1, x2, y2), p)
    if not keep_edges:
        return r, None
    # 그림용: measure_roi 안쪽과 같은 계산을 한 번 더 해서 엣지 위치를 꺼낸다
    bh = y2 - y1; m = int(bh * p.vertical_margin)
    my1 = max(0, y1 - m); my2 = min(img.shape[0], y2 + m)
    gray = cv2.cvtColor(img[my1:my2, x1:x2], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
    xs = np.linspace(int(blur.shape[1] * p.x_start_ratio), int(blur.shape[1] * p.x_end_ratio), p.n_points).astype(int)
    e = []
    for x in xs:
        o = sc.find_outer_edge(blur, x, p)
        if o: e.append(((x1 + x) / scale, (my1 + o[0]) / scale, (my1 + o[1]) / scale))
    return r, e

if __name__ == '__main__':
    # 질문 1: YOLO 를 10-08 사진에 그대로 (원본 엔진처럼 사진 전체, imgsz 640)
    yolo = []
    for rw in rows[::10]:          # 묶음마다 첫 장
        box, conf = sc.detect_box(os.path.join(D, rw['파일']))
        yolo.append((rw['파일'], rw['시편'], box, round(conf, 3)))
        print('YOLO', rw['파일'], rw['시편'], box, round(conf, 3), flush=True)

    out = csv.writer(open(os.path.join(OUT, 'engine_rows.csv'), 'w', newline='', encoding='utf-8-sig'))
    out.writerow(['파일', '묶음', '시편', '역할', '실측mm', '방법', '샤프트px', '샤프트sd', '유효점', '핀px', '핀sd', '핀유효점'])
    edges_keep = {}
    for i, rw in enumerate(rows):
        img = cv2.imdecode(np.fromfile(os.path.join(D, rw['파일']), np.uint8), cv2.IMREAD_COLOR)
        box = (int(rw['x0']), Y1, int(rw['x1']), Y2)
        first = (i % 10 == 0)
        for name, (s, p) in VARIANTS.items():
            keep = first and name.startswith('A')
            r, e = run_box(img, box, p, s, keep)
            rp, _ = run_box(img, PIN, p, s)
            if keep: edges_keep[rw['파일']] = e
            f = lambda d, k: (d[k] / s if k in ('measured_pixel', 'std_pixel') else d[k]) if d.get('ok') else ''
            out.writerow([rw['파일'], rw['묶음'], rw['시편'], rw['역할'], rw['실측mm'], name,
                          f(r, 'measured_pixel'), f(r, 'std_pixel'), r.get('n_valid', ''),
                          f(rp, 'measured_pixel'), f(rp, 'std_pixel'), rp.get('n_valid', '')])
        print(i, rw['파일'], rw['시편'], flush=True)
    np.save(os.path.join(OUT, 'engine_edges_A.npy'), edges_keep, allow_pickle=True)
