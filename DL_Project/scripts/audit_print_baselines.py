import json
from pathlib import Path

models = ['yolov8s', 'faster_rcnn', 'mask_rcnn', 'rf_detr_nano']
for m in models:
    p = Path(f'results/trained/{m}/test_metrics.json')
    if p.exists():
        with open(p) as f:
            d = json.load(f)
            print(f"{m:15s} | mAP50: {d.get('mAP50', 0):.4f} | Prec: {d.get('precision', 0):.4f} | Rec: {d.get('recall', 0):.4f} | F1: {d.get('f1', 0):.4f} | Small AP: {d.get('small_ap', 0):.4f} | FPS: {d.get('fps', 0):.2f}")
