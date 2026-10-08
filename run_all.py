"""
run_all.py — Unified runner for all 4 CV hackathon models.
Generates YOLO-format labels from YOLOv8 predictions, then evaluates:
  1. YOLOv8s
  2. Faster R-CNN
  3. Mask R-CNN
  4. YOLOv8 + SAM (skipped if SAM weights not found)
"""

import os, time, json
import numpy as np
from PIL import Image
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG — all paths relative to this script
# ─────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "dataset")
IMGS_DIR     = os.path.join(DATASET_DIR, "images")
LABS_DIR     = os.path.join(DATASET_DIR, "labels")
YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, "yolov8s.pt")
SAM_WEIGHTS  = os.path.join(SCRIPT_DIR, "sam_vit_l_0b3195.pth")

IOU_THRESH = 0.5
SMALL_AREA = 32 * 32

os.makedirs(LABS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def yolo_to_xyxy(xc, yc, w, h, W, H):
    return np.array([(xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H])

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    xc = ((x1+x2)/2) / W
    yc = ((y1+y2)/2) / H
    w  = (x2-x1) / W
    h  = (y2-y1) / H
    return xc, yc, w, h

def iou(a, b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = (a[2]-a[0])*(a[3]-a[1])
    aB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aA + aB - inter + 1e-6)

def compute_metrics(TP, FP, FN, ious, small_tp, small_total, times):
    precision  = TP / (TP+FP+1e-6)
    recall     = TP / (TP+FN+1e-6)
    f1         = 2*precision*recall / (precision+recall+1e-6)
    map50      = precision * recall
    mean_iou   = float(np.mean(ious)) if ious else 0.0
    small_map  = small_tp / (small_total+1e-6)
    avg_time   = float(np.mean(times))
    return dict(mAP50=round(map50,4), Precision=round(precision,4),
                Recall=round(recall,4), F1=round(f1,4),
                SmallObjectMAP=round(small_map,4),
                MeanIoU=round(mean_iou,4),
                AvgInferenceTime_s=round(avg_time,4))

def print_metrics(name, m):
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    for k, v in m.items():
        print(f"  {k:<25} {v}")

# ─────────────────────────────────────────────
# STEP 1 — Collect images
# ─────────────────────────────────────────────

image_files = sorted([
    os.path.join(IMGS_DIR, f)
    for f in os.listdir(IMGS_DIR)
    if f.lower().endswith(('.jpg','.jpeg','.png'))
])

if not image_files:
    raise FileNotFoundError(f"No images found in {IMGS_DIR}")

print(f"Found {len(image_files)} images: {[os.path.basename(f) for f in image_files]}")

# ─────────────────────────────────────────────
# STEP 2 — Generate YOLO labels using YOLOv8
#           (treat predictions as pseudo-GT)
# ─────────────────────────────────────────────

print("\n[INFO] Generating pseudo ground-truth labels with YOLOv8s...")
from ultralytics import YOLO
yolo_model = YOLO(YOLO_WEIGHTS)

label_files = []
for img_path in image_files:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    res  = yolo_model(img_path, verbose=False)[0]
    lab_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    lab_path = os.path.join(LABS_DIR, lab_name)
    with open(lab_path, "w") as f:
        for cls, box in zip(res.boxes.cls.cpu().numpy(),
                             res.boxes.xyxy.cpu().numpy()):
            xc, yc, w, h = xyxy_to_yolo(*box, W, H)
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    label_files.append(lab_path)
    print(f"  {os.path.basename(img_path)}: {len(res.boxes)} detections → label saved")

# ─────────────────────────────────────────────
# STEP 3 — Load GT for evaluation loop
# ─────────────────────────────────────────────

def load_gt(lab_path, W, H):
    gt = []
    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            c, xc, yc, w, h = map(float, parts[:5])
            box  = yolo_to_xyxy(xc, yc, w, h, W, H)
            area = (box[2]-box[0]) * (box[3]-box[1])
            gt.append((int(c), box, area))
    return gt

def eval_preds(gt, preds, TP, FP, FN, ious, small_tp, small_total):
    matched = set()
    for gc, gbox, garea in gt:
        best = 0; idx = -1
        for j, (pc, pbox) in enumerate(preds):
            v = iou(gbox, pbox)
            if v > best:
                best = v; idx = j
        if best >= IOU_THRESH:
            TP += 1; matched.add(idx); ious.append(best)
            if garea < SMALL_AREA: small_tp += 1
        else:
            FN += 1
        if garea < SMALL_AREA: small_total += 1
    FP += len(preds) - len(matched)
    return TP, FP, FN, ious, small_tp, small_total

# ─────────────────────────────────────────────
# MODEL 1 — YOLOv8s
# ─────────────────────────────────────────────

print("\n\n[MODEL 1] Running YOLOv8s...")
TP=FP=FN=0; ious=[]; times=[]; small_tp=small_total=0

for img_path, lab_path in zip(image_files, label_files):
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    gt  = load_gt(lab_path, W, H)
    t0  = time.time()
    res = yolo_model(img_path, verbose=False)[0]
    times.append(time.time()-t0)
    preds = [(int(c), b) for c, b in zip(
        res.boxes.cls.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
    TP,FP,FN,ious,small_tp,small_total = eval_preds(gt,preds,TP,FP,FN,ious,small_tp,small_total)

m1 = compute_metrics(TP,FP,FN,ious,small_tp,small_total,times)
print_metrics("YOLOv8s", m1)

# ─────────────────────────────────────────────
# MODEL 2 — Faster R-CNN
# ─────────────────────────────────────────────

print("\n\n[MODEL 2] Running Faster R-CNN (ResNet50-FPN)...")
import torch, torchvision
from torchvision.transforms import functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"
frcnn  = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
frcnn.eval().to(device)

TP=FP=FN=0; ious=[]; times=[]; small_tp=small_total=0

for img_path, lab_path in zip(image_files, label_files):
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    gt  = load_gt(lab_path, W, H)
    tensor = TF.to_tensor(img).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = frcnn([tensor])[0]
    times.append(time.time()-t0)
    preds = list(zip(out["labels"].cpu().numpy(), out["boxes"].cpu().numpy()))
    TP,FP,FN,ious,small_tp,small_total = eval_preds(gt,preds,TP,FP,FN,ious,small_tp,small_total)

m2 = compute_metrics(TP,FP,FN,ious,small_tp,small_total,times)
print_metrics("Faster R-CNN (ResNet50-FPN)", m2)

# ─────────────────────────────────────────────
# MODEL 3 — Mask R-CNN
# ─────────────────────────────────────────────

print("\n\n[MODEL 3] Running Mask R-CNN (ResNet50-FPN)...")
mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
mrcnn.eval().to(device)

TP=FP=FN=0; ious=[]; times=[]; small_tp=small_total=0

for img_path, lab_path in zip(image_files, label_files):
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    gt  = load_gt(lab_path, W, H)
    tensor = TF.to_tensor(img).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = mrcnn([tensor])[0]
    times.append(time.time()-t0)
    preds = list(zip(out["labels"].cpu().numpy(), out["boxes"].cpu().numpy()))
    TP,FP,FN,ious,small_tp,small_total = eval_preds(gt,preds,TP,FP,FN,ious,small_tp,small_total)

m3 = compute_metrics(TP,FP,FN,ious,small_tp,small_total,times)
print_metrics("Mask R-CNN (ResNet50-FPN)", m3)

# ─────────────────────────────────────────────
# MODEL 4 — YOLOv8 + SAM  (optional)
# ─────────────────────────────────────────────

if os.path.exists(SAM_WEIGHTS):
    print("\n\n[MODEL 4] Running YOLOv8 + SAM (ViT-L)...")
    from segment_anything import sam_model_registry, SamPredictor
    sam    = sam_model_registry["vit_l"](checkpoint=SAM_WEIGHTS)
    predictor = SamPredictor(sam)

    TP=FP=FN=0; ious=[]; times=[]; small_tp=small_total=0

    for img_path, lab_path in zip(image_files, label_files):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W   = img_np.shape[:2]
        gt     = load_gt(lab_path, W, H)
        t0     = time.time()
        det    = yolo_model(img_path, verbose=False)[0]
        boxes  = det.boxes.xyxy.cpu().numpy()
        predictor.set_image(img_np)
        sam_boxes = []
        for b in boxes:
            masks, _, _ = predictor.predict(box=b, multimask_output=False)
            m = masks[0]
            ys, xs = np.where(m)
            if len(xs) == 0: continue
            sam_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        times.append(time.time()-t0)
        preds = [(0, np.array(b)) for b in sam_boxes]
        TP,FP,FN,ious,small_tp,small_total = eval_preds(gt,preds,TP,FP,FN,ious,small_tp,small_total)

    m4 = compute_metrics(TP,FP,FN,ious,small_tp,small_total,times)
    print_metrics("YOLOv8 + SAM (ViT-L)", m4)
else:
    print(f"\n[MODEL 4] SKIPPED — SAM weights not found at:\n  {SAM_WEIGHTS}")
    print("  To enable: download sam_vit_l_0b3195.pth (~2.4 GB) from:")
    print("  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
    m4 = None

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n\n" + "="*70)
print("  COMPARISON SUMMARY")
print("="*70)
header = f"{'Metric':<25} {'YOLOv8s':>12} {'FasterRCNN':>12} {'MaskRCNN':>12}"
if m4: header += f" {'YOLO+SAM':>12}"
print(header)
print("-"*70)

keys = ["mAP50","Precision","Recall","F1","SmallObjectMAP","MeanIoU","AvgInferenceTime_s"]
for k in keys:
    row = f"{k:<25} {m1[k]:>12} {m2[k]:>12} {m3[k]:>12}"
    if m4: row += f" {m4[k]:>12}"
    print(row)

print("="*70)
print("\n[DONE] All models evaluated.")
