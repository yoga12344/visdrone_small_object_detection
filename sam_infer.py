import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ---------------- paths ----------------

images = [
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04200_d_0000014.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04674_d_0000016.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04925_d_0000017.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_04999_d_0000018.jpg",
r"C:\Users\bomma\Downloads\dl hack\compare\0000076_05079_d_0000019.jpg",
]

labels = [
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04200_d_0000014.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04674_d_0000016.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04925_d_0000017.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_04999_d_0000018.txt",
r"C:\Users\bomma\Downloads\dl hack\compare\tmp_compare_dataset\labels\val\0000076_05079_d_0000019.txt",
]

sam_ckpt = r"C:\Users\bomma\Downloads\sam_vit_l_0b3195.pth"
yolo_weights = r"C:\Users\bomma\Downloads\yolov8s.pt"

IOU_THRESH = 0.5
SMALL_AREA = 32*32

# ---------------- helpers ----------------

def yolo_to_xyxy(xc,yc,w,h,W,H):
    return np.array([(xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H])

def box_iou(a,b):
    xA=max(a[0],b[0]);yA=max(a[1],b[1])
    xB=min(a[2],b[2]);yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

# ---------------- load models ----------------

yolo = YOLO(yolo_weights)
sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)
predictor = SamPredictor(sam)

TP=FP=FN=0
ious=[]
small_tp=small_total=0
times=[]

# ---------------- main loop ----------------

for img_path, lab_path in zip(images, labels):

    img = np.array(Image.open(img_path))
    H,W = img.shape[:2]

    gt=[]
    with open(lab_path) as f:
        for l in f:
            _,xc,yc,w,h = map(float,l.split())
            b=yolo_to_xyxy(xc,yc,w,h,W,H)
            area=(b[2]-b[0])*(b[3]-b[1])
            gt.append((b,area))

    t0=time.time()
    det=yolo(img_path)[0]
    boxes=det.boxes.xyxy.cpu().numpy()

    predictor.set_image(img)

    sam_boxes=[]
    for b in boxes:
        masks,_,_=predictor.predict(box=b,multimask_output=False)
        m=masks[0]
        ys,xs=np.where(m)
        if len(xs)==0: continue
        sam_boxes.append([xs.min(),ys.min(),xs.max(),ys.max()])

    t1=time.time()
    times.append(t1-t0)

    matched=set()

    for gbox,garea in gt:
        best=0;idx=-1
        for j,p in enumerate(sam_boxes):
            v=box_iou(gbox,p)
            if v>best:
                best=v;idx=j

        if best>=IOU_THRESH:
            TP+=1
            matched.add(idx)
            ious.append(best)
            if garea<SMALL_AREA: small_tp+=1
        else:
            FN+=1

        if garea<SMALL_AREA: small_total+=1

    FP+=len(sam_boxes)-len(matched)

# ---------------- metrics ----------------

precision=TP/(TP+FP+1e-6)
recall=TP/(TP+FN+1e-6)
f1=2*precision*recall/(precision+recall+1e-6)
map50=precision*recall
mean_iou=np.mean(ious) if ious else 0
small_map=small_tp/(small_total+1e-6)
avg_time=np.mean(times)

print("\n===== SAM + YOLO METRICS =====")
print("mAP@0.5:",round(map50,4))
print("Precision:",round(precision,4))
print("Recall:",round(recall,4))
print("F1:",round(f1,4))
print("Small Object mAP:",round(small_map,4))
print("Mean IoU:",round(mean_iou,4))
print("Avg Inference Time:",round(avg_time,4))
