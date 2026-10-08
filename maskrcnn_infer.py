import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import time
from collections import defaultdict

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

IOU_THRESH = 0.5
SMALL_AREA = 32*32

# ---------------- helpers ----------------

def yolo_to_xyxy(xc,yc,w,h,W,H):
    return np.array([(xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H])

def iou(a,b):
    xA=max(a[0],b[0]);yA=max(a[1],b[1])
    xB=min(a[2],b[2]);yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

# ---------------- model ----------------

device="cpu"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.eval().to(device)

TP=FP=FN=0
ious=[]
times=[]
small_tp=small_total=0
confusion=defaultdict(lambda:defaultdict(int))

# ---------------- loop ----------------

for img_path,lab_path in zip(images,labels):

    img = Image.open(img_path).convert("RGB")
    W,H = img.size
    tensor = F.to_tensor(img).to(device)

    gt=[]
    with open(lab_path) as f:
        for l in f:
            c,xc,yc,w,h = map(float,l.split())
            box=yolo_to_xyxy(xc,yc,w,h,W,H)
            area=(box[2]-box[0])*(box[3]-box[1])
            gt.append((int(c),box,area))

    t0=time.time()
    out=model([tensor])[0]
    t1=time.time()
    times.append(t1-t0)

    preds=list(zip(
    out["labels"].detach().cpu().numpy(),
    out["boxes"].detach().cpu().numpy()
    ))


    matched=set()

    for gc,gbox,garea in gt:
        best=0;idx=-1
        for j,(pc,pbox) in enumerate(preds):
            v=iou(gbox,pbox)
            if v>best:
                best=v;idx=j

        if best>=IOU_THRESH:
            TP+=1
            matched.add(idx)
            ious.append(best)
            pc,_=preds[idx]
            confusion[gc][pc]+=1
            if garea<SMALL_AREA: small_tp+=1
        else:
            FN+=1

        if garea<SMALL_AREA: small_total+=1

    FP+=len(preds)-len(matched)

# ---------------- metrics ----------------

precision=TP/(TP+FP+1e-6)
recall=TP/(TP+FN+1e-6)
f1=2*precision*recall/(precision+recall+1e-6)
map50=precision*recall
mean_iou=np.mean(ious) if ious else 0
small_map=small_tp/(small_total+1e-6)
avg_time=np.mean(times)

print("\n===== MASK R-CNN METRICS =====")
print("mAP@0.5:",round(map50,4))
print("Precision:",round(precision,4))
print("Recall:",round(recall,4))
print("F1:",round(f1,4))
print("Small Object mAP:",round(small_map,4))
print("Mean IoU:",round(mean_iou,4))
print("Avg Inference Time:",round(avg_time,4))

print("\nConfusion Matrix (GT->Pred):")
for k in confusion:
    print(k,dict(confusion[k]))
