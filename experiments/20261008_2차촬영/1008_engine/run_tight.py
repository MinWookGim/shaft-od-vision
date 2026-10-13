# 옵션 수정 실험: 원본 규칙(가장 바깥 쌍)은 그대로 두고, 측정 박스를 위아래로 넓히지 않는다(vertical_margin=0)
# 넓히면 위쪽 검은 가림막 경계·아래 배경 경계가 박스에 들어와 "가장 바깥 쌍"이 그걸 집는다(그림 edges_overlay.png)
# 박스를 안 넓히면 허용 지름범위가 ROI 비율로는 맞지 않아서 박스 높이 기준(dia_basis=box, 0.55~0.95)으로 둔다
import os, csv, numpy as np, cv2
import run_engine as R
sc = R.sc
V = {'E_원본규칙_박스안넓힘': sc.Params(vertical_margin=0.0, dia_basis='box'),
     'F_가장센쌍_박스안넓힘': sc.Params(vertical_margin=0.0, dia_basis='box', edge_select='strongest')}
w = csv.writer(open(os.path.join(R.OUT, 'engine_rows_tight.csv'), 'w', newline='', encoding='utf-8-sig'))
w.writerow(['파일', '묶음', '시편', '역할', '실측mm', '방법', '샤프트px', '샤프트sd', '유효점', '핀px', '핀sd', '핀유효점'])
for i, rw in enumerate(R.rows):
    img = cv2.imdecode(np.fromfile(os.path.join(R.D, rw['파일']), np.uint8), cv2.IMREAD_COLOR)
    box = (int(rw['x0']), R.Y1, int(rw['x1']), R.Y2)
    for name, p in V.items():
        r, _ = R.run_box(img, box, p, 1.0); rp, _ = R.run_box(img, R.PIN, p, 1.0)
        g = lambda d, k: d.get(k, '') if d.get('ok') else ''
        w.writerow([rw['파일'], rw['묶음'], rw['시편'], rw['역할'], rw['실측mm'], name,
                    g(r, 'measured_pixel'), g(r, 'std_pixel'), r.get('n_valid', ''), g(rp, 'measured_pixel'), g(rp, 'std_pixel'), rp.get('n_valid', '')])
print('ok')
