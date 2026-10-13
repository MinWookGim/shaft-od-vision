# 엔진이 잡은 엣지를 사진 위에, 그리고 밝기 단면 위에 그린다 (멘토: 경계를 제대로 잡는지 이미지로 먼저 확인)
# 그림 1: 한 장의 베어링부 확대 + 규칙별로 잡은 위·아래 엣지
# 그림 2: 한 열의 밝기 단면과 변화율(gradient) + 각 규칙이 고른 자리
import os, numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumSquare' in f and 'Bold' not in f and 'Extra' not in f: fm.fontManager.addfont(f)
plt.rcParams['font.family'] = ['NanumSquare', 'DejaVu Sans']
import run_engine as R
sc = R.sc

RULES = [('원본(가장 바깥 쌍)', sc.Params(), '#d62728'), ('가장 센 쌍', sc.Params(edge_select='strongest'), '#1f77b4'),
         ('중앙값 쌍', sc.Params(edge_select='median'), '#ff7f0e')]

def pick(nrow):
    rw = R.rows[nrow]
    img = cv2.imdecode(np.fromfile(os.path.join(R.D, rw['파일']), np.uint8), cv2.IMREAD_COLOR)
    return rw, img

def fig_overlay(nrows, name):
    fig, axes = plt.subplots(1, len(nrows), figsize=(4.2 * len(nrows), 7))
    for ax, n in zip(np.atleast_1d(axes), nrows):
        rw, img = pick(n)
        x0, x1 = int(rw['x0']), int(rw['x1'])
        box = (x0, R.Y1, x1, R.Y2)
        ya, yb = 3300, 4600
        crop = cv2.cvtColor(img[ya:yb, x0 - 60:x1 + 60], cv2.COLOR_BGR2RGB)
        ax.imshow(crop, extent=(x0 - 60, x1 + 60, yb, ya), aspect='auto')
        for lab, p, c in RULES:
            _, e = R.run_box(img, box, p, 1.0, True)
            e = np.array(e)
            ax.plot(e[:, 0], e[:, 1], '.', ms=5, color=c, label=lab)
            ax.plot(e[:, 0], e[:, 2], '.', ms=5, color=c)
        ax.axhspan(R.Y1, R.Y2, color='lime', alpha=0.06)
        ax.set_title(f"{rw['시편']} ({rw['파일'][9:15]})\n내 방법 {float(rw['샤프트px']):.1f}px", fontsize=10)
        ax.set_xticks([])
    np.atleast_1d(axes)[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('엔진이 고른 위·아래 엣지 (점 = 측정선 31개 각각). 연두 띠 = 측정 박스', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(R.OUT, name), dpi=110); plt.close(fig)

def fig_profile(n, name):
    rw, img = pick(n)
    x0, x1 = int(rw['x0']), int(rw['x1'])
    p = sc.Params()
    bh = R.Y2 - R.Y1; m = int(bh * p.vertical_margin); my1 = R.Y1 - m; my2 = R.Y2 + m
    gray = cv2.cvtColor(img[my1:my2, x0:x1], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    xs = np.linspace(int(blur.shape[1] * p.x_start_ratio), int(blur.shape[1] * p.x_end_ratio), p.n_points).astype(int)
    x = xs[len(xs) // 2]
    prof = blur[:, x - 2:x + 3].mean(1); grad = np.abs(np.gradient(prof)); yy = np.arange(len(prof)) + my1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(yy, prof, color='k', lw=1); a1.set_ylabel('밝기')
    a2.plot(yy, grad, color='gray', lw=1); a2.set_ylabel('밝기 변화율(절댓값)')
    h = len(prof)
    a2.axhline(np.percentile(grad[:int(h * .46)], p.grad_percentile), color='gray', ls=':', lw=.8)
    for lab, pp, c in RULES:
        o = sc.find_outer_edge(blur, x, pp)
        if o:
            for v in o[:2]:
                a1.axvline(v + my1, color=c, lw=1.2); a2.axvline(v + my1, color=c, lw=1.2)
            a1.plot([], [], color=c, label=f'{lab}: {o[2]:.1f}px')
    # 내 방법: 배경 밝기와 어두운 쪽 중간값을 가로지르는 곳
    # 내 방법은 측정 박스(Y1~Y2) 안에서만 찾는다. 박스 밖 위쪽 어두운 천장을 넣으면 엉뚱한 데를 잡는다
    col = blur[R.Y1 - my1:R.Y2 - my1, x].astype(float)
    t, b = R.edges(col, (np.percentile(col, 95) + np.percentile(col, 5)) / 2)
    for v in (t, b): a1.axvline(v + R.Y1, color='green', ls='--', lw=1)
    a1.plot([], [], color='green', ls='--', label=f'50% 밝기(내 방법): {b - t:.1f}px')
    a1.legend(fontsize=9, loc='upper right')
    a1.set_title(f"{rw['시편']} {rw['파일']} 가운데 측정선 하나의 세로 단면 (왼쪽=위, 오른쪽=아래)", fontsize=11)
    a2.set_xlabel('사진 세로 위치(px)')
    fig.tight_layout(); fig.savefig(os.path.join(R.OUT, name), dpi=110); plt.close(fig)

if __name__ == '__main__':
    fig_overlay([0, 20, 40, 60, 100], 'edges_overlay.png')    # NO1(P1), NO3, NO5, NO7, NG1
    fig_profile(40, 'edge_profile_NO5.png')
    fig_profile(0, 'edge_profile_NO1.png')
    print('ok')
