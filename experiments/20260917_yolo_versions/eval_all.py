# 버전별 학습 모델(+기존 best.pt, +앙상블)을 161장 2992 축소본과 6008 원본에서 평가
# 성공 기준(사전 설정): 검출 161/161 · val 시편 IoU 중앙값 ≥ 0.8 · 박스 높이 시편내 sd · 측정 px 시편내 sd · 6008 원본에서 붕괴(비율 밖) 장수
import json, sys, csv, numpy as np, cv2, pathlib
from statistics import median
sys.path.insert(0, "src")
EXP = "src"; sys.path.insert(0, EXP)
import shaft_core as core; from shaft_core import Params
from ultralytics import YOLO
R = pathlib.Path(__file__).parent
GT = json.load(open(R/"gt_boxes.json"))
man = list(csv.DictReader(open("experiments/results/20260916/manifest_20260916.csv", encoding="utf-8-sig")))
spec_of = {r["파일"]: r["시편"] for r in man}
VAL = {"S07","S08","S09","S10","N02","N03"}
ORIG = pathlib.Path("src/photos/20260916_원전기계")
models = {p.name: str(p/"weights/best.pt") for p in sorted((R/"runs/detect/runs").glob("*")) if (p/"weights/best.pt").exists()}
models["기존 best.pt(7월,26n)"] = "best_7월.pt"

def iou(a, b):
    ix = max(0, min(a[2],b[2]) - max(a[0],b[0])); iy = max(0, min(a[3],b[3]) - max(a[1],b[1]))
    inter = ix*iy; return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

def predict(model, path, imgsz=640):
    r = model.predict(str(path), imgsz=imgsz, conf=0.05, verbose=False)[0]
    if len(r.boxes) == 0: return None, 0.0
    i = int(r.boxes.conf.argmax()); return r.boxes.xyxy[i].tolist(), float(r.boxes.conf[i])

files = sorted(GT)
preds = {}   # name -> file -> (box, conf)
for name, w in models.items():
    m = YOLO(w); preds[name] = {}
    for fn in files:
        preds[name][fn] = predict(m, R/"sm161"/fn)
    print("predicted", name, flush=True)
# 앙상블: 새로 학습한 n 모델들의 박스를 conf 가중 평균 (검출된 것만)
ens_members = [n for n in models if n.endswith("n") and "기존" not in n]
preds["앙상블(n 5종 conf가중)"] = {}
for fn in files:
    bs = [(b, c) for n in ens_members for b, c in [preds[n][fn]] if b]
    if bs:
        wsum = sum(c for _, c in bs); box = [sum(b[i]*c for b, c in bs)/wsum for i in range(4)]
        preds["앙상블(n 5종 conf가중)"][fn] = (box, wsum/len(ens_members))
    else: preds["앙상블(n 5종 conf가중)"][fn] = (None, 0.0)

print("\n=== 2992 축소본 (val 시편 = 학습에 안 쓴 6시편 70장)")
print(f"{'모델':28s} 검출 val  IoU중앙 IoU최소  conf중앙  박스높이sd(시편내,px)  측정px sd(시편내)  측정 실패")
rows_out = []
for name, pr in preds.items():
    det = [fn for fn in files if pr[fn][0] and spec_of[fn] in VAL]
    ious = [iou(pr[fn][0], GT[fn]) for fn in det]; confs = [pr[fn][1] for fn in det]
    hsd, msd, fail = [], [], 0
    for s in sorted(VAL):
        fs = [fn for fn in det if spec_of[fn] == s]
        if len(fs) > 2:
            hsd.append(np.std([pr[fn][0][3]-pr[fn][0][1] for fn in fs], ddof=1))
            px = []
            for fn in fs:
                img = core.imread(str(R/"sm161"/fn)); r = core.measure_roi(img, tuple(int(v) for v in pr[fn][0]), Params())
                if r["ok"]: px.append(r["measured_pixel"])
                else: fail += 1
            if len(px) > 2: msd.append(np.std(px, ddof=1))
    n_val = sum(1 for fn in files if spec_of[fn] in VAL)
    print(f"{name:28s} {len(det):3d}/{n_val}  {median(ious) if ious else 0:6.3f}  {min(ious) if ious else 0:6.3f}  {median(confs) if confs else 0:6.3f}   {np.mean(hsd) if hsd else float('nan'):8.2f}            {np.mean(msd) if msd else float('nan'):6.3f}        {fail}")
    rows_out.append(dict(model=name, det=len(det), n=n_val, iou_med=median(ious) if ious else 0, iou_min=min(ious) if ious else 0, conf_med=median(confs) if confs else 0, box_h_sd=float(np.mean(hsd)) if hsd else None, meas_sd=float(np.mean(msd)) if msd else None, meas_fail=fail))

print("\n=== 6008 원본 직접 투입 (S01 14장) — 박스 높이 비율(지름/박스높이) 0.65~0.85 밖 = 붕괴 후보")
for name, w in models.items():
    m = YOLO(w); bad = 0; hs = []
    for fn in [f for f in files if spec_of[f] == "S01"]:
        b, c = predict(m, ORIG/fn)
        if not b: bad += 1; continue
        h = b[3]-b[1]; hs.append(h)
        gt_h = (GT[fn][3]-GT[fn][1]) * 6008/2992
        if abs(h/gt_h - 1) > 0.15: bad += 1
    print(f"{name:28s} 박스높이 이상/미검출 {bad}/14   높이 중앙 {median(hs) if hs else 0:.0f} (기대 {gt_h:.0f})")
json.dump(dict(rows=rows_out, preds={n:{fn:list(v) for fn,v in pr.items()} for n,pr in preds.items()}), open(R/"eval_result.json","w"), default=float)
