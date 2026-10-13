# 실험 1: 멘토 증류 이상 탐지를 10-08 샤프트 사진에 그대로 적용
# 규칙(돌리기 전에 고정)
#   정상 = 묶음 1~10, 13 (NO1~NO10, NO1 재촬영). NG = 묶음 11, 12
#   2겹: A = 홀수 묶음으로 학습 -> 짝수 묶음 + NG 로 시험, B = 반대. 같은 시편이 학습·시험에 섞이지 않게
#   보는 해상도 2가지: 넓게 = 1024px 잘라 256 (1px=약 148um), 좁게 = 위 경계 중심 512px 잘라 256 (약 74um)
# 시험 3가지
#   T1 진짜 NG(실측 규격 하한 아래) 를 정상과 가르나 -> AUC (0.5 = 동전 던지기)
#   T2 초점 구간(F1/F2 vs F3)에 따라 점수가 갈리나
#   T3 인공 결함을 시험 정상 사진에 그려 넣고(원본 해상도에서 그린 뒤 줄임) 깨끗한 같은 사진과 가르나
#       위 경계 패임(notch)·돌기(burr) 반지름 0.1/0.2/0.5/1.0mm, 표면 긁힘 폭 0.05/0.1/0.2mm 길이 3mm
#       + 히트맵 최대점이 결함 자리(256 기준 16px 안)에 오나
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); import kdad
dev = 'cuda'; torch.manual_seed(0); np.random.seed(0)
Z = np.load(os.path.join(ROOT, '_data', 'ad_crops.npz'), allow_pickle=True)
CROPS, INFO = Z['crops'], Z['info']
GROUP = np.array([int(i[1]) for i in INFO]); FOCUS = np.array([i[4] for i in INFO])
HPX = np.array([float(i[5]) for i in INFO]); BG = np.array([float(i[6]) for i in INFO])
UM = 36.95                                # 1px 약 37um (10-08)
NG = [11, 12]; NORMAL = [g for g in sorted(set(GROUP)) if g not in NG]
VIEWS = {'넓게_148um': (1024, 'mid'), '좁게_74um': (512, 'top')}
EPOCHS, LAMDA, OUT = int(os.environ.get('EPOCHS', 300)), 0.5, 256

def center(i, view):
    size, where = VIEWS[view]
    cy = 640 if where == 'mid' else int(round(640 - HPX[i] / 2))
    return 640, cy, size

def cut(img, cx, cy, size, dx=0, dy=0):
    a = img[cy - size // 2 + dy: cy + size // 2 + dy, cx - size // 2 + dx: cx + size // 2 + dx].astype(np.float32) / 255
    t = torch.from_numpy(a)[None, None]
    return F.interpolate(t, size=(OUT, OUT), mode='area')[0].repeat(3, 1, 1)

def batch_train(idx, view, bs=16):
    def gen():
        p = np.random.permutation(idx)
        for k in range(0, len(p), bs):
            xs = []
            for i in p[k:k + bs]:
                cx, cy, size = center(i, view); j = size // 16
                x = cut(CROPS[i], cx, cy, size, np.random.randint(-j, j + 1), np.random.randint(-j, j + 1))
                xs.append((x * np.random.uniform(.95, 1.05)).clamp(0, 1))
            yield torch.stack(xs)
    return gen

def defect(img, i, kind, mm, rng):
    """원본 해상도 1280 자른 사진에 결함을 그린다. 돌려주는 값: 그린 사진, 결함 중심(x, y)"""
    g = img.astype(np.float32).copy(); top = 640 - HPX[i] / 2
    x0 = 640 + rng.uniform(-80, 80); yy, xx = np.mgrid[0:1280, 0:1280]
    dark = np.percentile(g[int(top) + 20:int(top) + 120, 560:720], 50)
    if kind in ('notch', 'burr'):
        r = mm * 1000 / UM; d = (xx - x0) ** 2 + (yy - top) ** 2 < r ** 2
        m = d & (yy > top) if kind == 'notch' else d & (yy < top)
        g[m] = BG[i] if kind == 'notch' else dark
        c = (x0, top)
    else:                                  # 긁힘: 표면 위(위 경계에서 0.15 지름 아래) 3mm 사선, 주변보다 80 밝게
        w = mm * 1000 / UM / 2; L = 3000 / UM; y0 = top + 0.15 * HPX[i]; ang = np.deg2rad(10)
        u = (xx - x0) * np.cos(ang) + (yy - y0) * np.sin(ang); v = -(xx - x0) * np.sin(ang) + (yy - y0) * np.cos(ang)
        m = (np.abs(u) < L / 2) & (np.abs(v) < max(w, .5))
        g[m] = np.clip(g[m] + 80, 0, 255); c = (x0, y0)
    return g.clip(0, 255).astype(np.uint8), c

DEFECTS = [('notch', m) for m in (.1, .2, .5, 1.)] + [('burr', m) for m in (.1, .2, .5, 1.)] + [('scratch', m) for m in (.05, .1, .2)]

def run(view, log):
    res = {'T1': [], 'T2': [], 'T3': {}}
    for fold, train_par in (('A', 1), ('B', 0)):
        tr = np.where(np.isin(GROUP, [g for g in NORMAL if g % 2 == train_par]))[0]
        te_n = np.where(np.isin(GROUP, [g for g in NORMAL if g % 2 != train_par]))[0]
        te_ng = np.where(np.isin(GROUP, NG))[0]
        T = kdad.Teacher().to(dev); S = kdad.Student().to(dev)
        kdad.train(S, T, batch_train(tr, view), EPOCHS, LAMDA, dev, log=lambda s: log(f'[{view} {fold}] {s}'))
        def sc(idx, imgs=None):
            xs = torch.stack([cut(CROPS[i] if imgs is None else imgs[k], *center(i, view)) for k, i in enumerate(idx)])
            return kdad.score(S, T, xs, LAMDA, dev)
        sn, sg = sc(te_n), sc(te_ng)
        auc = roc_auc_score([0] * len(sn) + [1] * len(sg), np.r_[sn, sg])
        res['T1'].append(dict(fold=fold, auc=round(auc, 3),
                              정상_묶음별={int(g): round(float(sn[GROUP[te_n] == g].mean()), 4) for g in sorted(set(GROUP[te_n]))},
                              NG_묶음별={int(g): round(float(sg[GROUP[te_ng] == g].mean()), 4) for g in NG}))
        f3 = FOCUS[te_n] == 'F3'
        res['T2'].append(dict(fold=fold, auc_F3_vs_F12=round(roc_auc_score(f3, sn), 3) if 0 < f3.sum() < len(f3) else None))
        rng = np.random.default_rng(1)
        for kind, mm in DEFECTS:
            imgs, cs = zip(*[defect(CROPS[i], i, kind, mm, rng) for i in te_n])
            sd = sc(te_n, imgs)
            a = roc_auc_score([0] * len(sn) + [1] * len(sd), np.r_[sn, sd])
            # 히트맵 최대점이 결함 자리에 오나
            cx, cy, size = center(te_n[0], view); hits = 0
            xs = torch.stack([cut(imgs[k], *center(i, view)) for k, i in enumerate(te_n)])
            hm = np.concatenate([kdad.heatmap(S, T, xs[k:k + 16], LAMDA, dev) for k in range(0, len(xs), 16)])
            for k, i in enumerate(te_n):
                cx, cy, size = center(i, view); s = OUT / size
                tx, ty = (cs[k][0] - (cx - size // 2)) * s, (cs[k][1] - (cy - size // 2)) * s
                py, px = np.unravel_index(hm[k].argmax(), hm[k].shape)
                hits += np.hypot(px - tx, py - ty) < 16
                if k == 0: t0 = (tx, ty)      # 그림용 (10-11 1차 실행은 마지막 장 자리를 저장하는 버그가 있었다)
            if fold == 'A' and kind != 'scratch' and mm == .5 or fold == 'A' and kind == 'scratch' and mm == .2:
                np.save(os.path.join(ROOT, '_out', f'ad_hm_{view}_{kind}_{mm}.npy'),
                        dict(img=xs[0][0].numpy(), hm=hm[0], t=t0), allow_pickle=True)
            res['T3'].setdefault(f'{kind}_{mm}mm', []).append(dict(fold=fold, auc=round(a, 3), 위치맞춤=f'{hits}/{len(te_n)}'))
            log(f'[{view} {fold}] {kind} {mm}mm AUC {a:.3f} 위치 {hits}/{len(te_n)}')
        log(f'[{view} {fold}] T1 NG AUC {auc:.3f}  T2 {res["T2"][-1]}')
    return res

if __name__ == '__main__':
    lf = open(os.path.join(ROOT, '_out', 'ad_run.log'), 'a')
    log = lambda s: (print(s, flush=True), lf.write(s + '\n'), lf.flush())
    out = {v: run(v, log) for v in VIEWS}
    json.dump(out, open(os.path.join(ROOT, '_out', 'ad_result.json'), 'w'), ensure_ascii=False, indent=1)
    log('done')
