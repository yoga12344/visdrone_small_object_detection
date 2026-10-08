from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
from collections import defaultdict

# -------------------------
# Your paths
# -------------------------

images = [
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04200_d_0000014.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04674_d_0000016.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04925_d_0000017.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04999_d_0000018.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_05079_d_0000019.jpg",
]

raw_labels = [
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04200_d_0000014.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04674_d_0000016.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04925_d_0000017.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04999_d_0000018.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_05079_d_0000019.txt",
]

weights = r"C:\Users\bomma\Downloads\yolov8s.pt"

IOU_THRESH = 0.5
SMALL_AREA = 32 * 32

# -------------------------
# Helpers
# -------------------------

def yolo_to_xyxy(xc, yc, w, h, W, H):
    return np.array([
        (xc - w/2) * W,
        (yc - h/2) * H,
        (xc + w/2) * W,
        (yc + h/2) * H
    ])

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# -------------------------
# Load model
# -------------------------

model = YOLO(weights)

all_ious = []
TP = FP = FN = 0
small_tp = small_total = 0
confusion = defaultdict(lambda: defaultdict(int))
times = []

# -------------------------
# Main loop
# -------------------------

for img_path, lab_path in zip(images, raw_labels):

    W, H = Image.open(img_path).size

    # GT
    gt = []
    with open(lab_path) as f:
        for l in f:
            c, xc, yc, w, h = map(float, l.split())
            box = yolo_to_xyxy(xc, yc, w, h, W, H)
            area = (box[2]-box[0])*(box[3]-box[1])
            gt.append((int(c), box, area))

    # Prediction
    t0 = time.time()
    pred = model(img_path)[0]
    t1 = time.time()
    times.append(t1-t0)

    preds = [(int(c), b) for c,b in zip(
        pred.boxes.cls.cpu().numpy(),
        pred.boxes.xyxy.cpu().numpy()
    )]

    matched_gt = set()
    matched_pr = set()

    for i,(gc,gbox,garea) in enumerate(gt):
        best = 0
        best_j = -1
        for j,(pc,pbox) in enumerate(preds):
            v = iou(gbox, pbox)
            if v > best:
                best = v
                best_j = j

        if best >= IOU_THRESH:
            TP += 1
            matched_gt.add(i)
            matched_pr.add(best_j)
            all_ious.append(best)

            pc,_ = preds[best_j]
            confusion[gc][pc] += 1

            if garea < SMALL_AREA:
                small_tp += 1
        else:
            FN += 1

        if garea < SMALL_AREA:
            small_total += 1

    FP += len(preds) - len(matched_pr)

# -------------------------
# Metrics
# -------------------------

precision = TP / (TP+FP+1e-6)
recall = TP / (TP+FN+1e-6)
f1 = 2*precision*recall/(precision+recall+1e-6)
mean_iou = np.mean(all_ious) if all_ious else 0
map50 = precision * recall
small_map = small_tp / (small_total+1e-6)
avg_time = np.mean(times)

print("\n===== FINAL METRICS =====")
print("mAP@0.5:", round(map50,4))
print("Precision:", round(precision,4))
print("Recall:", round(recall,4))
print("F1:", round(f1,4))
print("Small Object mAP:", round(small_map,4))
print("Mean IoU:", round(mean_iou,4))
print("Avg Inference Time (s):", round(avg_time,4))

print("\nConfusion Matrix (GT -> Pred):")
for k in confusion:
    print(k, dict(confusion[k]))
