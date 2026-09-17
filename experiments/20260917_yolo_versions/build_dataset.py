# 161장 2992 축소본에 프로파일 기반 자동 라벨(측정부 박스)을 붙여 YOLO 데이터셋을 만든다
# 박스 규약은 기존 YOLO 박스에 맞춤: x = 칼라끝+7%L ~ 홈-15%L, y = 윗변-74 ~ 아랫변+74
# 분할은 시편 단위(누수 방지): train S01~S06+N01 (7시편 91장) / val S07~S10+N02+N03 (6시편 70장)
import json, numpy as np, pathlib, shutil, cv2
R = pathlib.Path(__file__).parent
feats = json.load(open("feats.json"))
P = json.load(open("profiles.json"))
X = np.array(P["xs"]); prof = P["prof"]
TRAIN = {"S01","S02","S03","S04","S05","S06","N01"}
W = H = 2992
for split in ("train","val"):
    (R/"ds"/split/"images").mkdir(parents=True, exist_ok=True); (R/"ds"/split/"labels").mkdir(parents=True, exist_ok=True)
gt = {}
for f in feats:
    fn, spec = f["file"], f["spec"]; L = f["xg"] - f["xs"]
    x1, x2 = f["xs"] + 0.07*L, f["xg"] - 0.15*L
    t = np.array(prof[fn]["top"], float); b = np.array(prof[fn]["bot"], float)
    m = (X >= x1) & (X <= x2)
    y1, y2 = np.nanmedian(t[m]) - 74, np.nanmedian(b[m]) + 74
    gt[fn] = [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
    split = "train" if spec in TRAIN else "val"
    shutil.copy(R/"sm161"/fn, R/"ds"/split/"images"/fn)
    cx, cy, w, h = (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H
    (R/"ds"/split/"labels"/fn.replace(".jpg",".txt")).write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
json.dump(gt, open(R/"gt_boxes.json","w"))
(R/"ds"/"data.yaml").write_text(f"path: {R/'ds'}\ntrain: train/images\nval: val/images\nnames:\n  0: measurement_roi\n")
print("train", len(list((R/"ds/train/images").glob("*.jpg"))), "val", len(list((R/"ds/val/images").glob("*.jpg"))))
# 라벨 확인용 1장
fn = "20260916_155448.jpg"; img = cv2.imread(str(R/"sm161"/fn)); x1,y1,x2,y2 = map(int, gt[fn])
cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 4); cv2.imwrite(str(R/"label_check_S07.jpg"), cv2.resize(img,(1000,1000)))
