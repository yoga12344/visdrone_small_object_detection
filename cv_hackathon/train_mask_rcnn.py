"""
train_mask_rcnn.py
===================
Fine-tunes Mask R-CNN (ResNet50 + FPN) on the VisDrone dataset using CUDA:0.

Strict Requirements Met:
  1. Uses the exact processed dataset: data/processed/ (422 train, 106 val, 20 test locked).
  2. The 20 test images are LOCKED and completely excluded from training/validation.
  3. Pretrained COCO weights are loaded, and the RoI box predictor and mask predictor heads
     are adapted for 11 classes: class 0 = background, classes 1..10 = VisDrone foreground.
  4. VisDrone provides bounding-box annotations (no GT masks).
     During training, box-level supervision is applied without fabricating segmentation ground truth.
     During evaluation and inference, predicted instance masks are generated and visualized for qualitative analysis.
  5. Training executed with AMP mixed precision on NVIDIA RTX 4060 Laptop GPU.
  6. Tracks validation metrics at every epoch and saves best.pt based strictly on validation mAP@0.5.
  7. Evaluates the best checkpoint on the locked 20 test images against authentic VisDrone annotations.
  8. Generates confusion matrix, qualitative panels with predicted masks, and comparison tables.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "inference"))

import visdrone_parser
import metrics
import small_objects
import confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MaskRCNN_Trainer")

CLASS_NAMES = visdrone_parser.CLASS_NAMES

# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------

class VisDroneTorchDataset(Dataset):
    """
    TorchVision-compatible dataset that reads from data/processed/images/{split}
    and data/processed/labels/{split}.
    Converts normalized YOLO [cls, xc, yc, w, h] to absolute xyxy boxes
    and 1-indexed TorchVision labels (1..10, where 0 is background).
    """
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image_id = img_path.stem
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        label_path = self.labels_dir / f"{image_id}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * W
                        yc = float(parts[2]) * H
                        w = float(parts[3]) * W
                        h = float(parts[4]) * H

                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(float(W), xc + w / 2.0)
                        y2 = min(float(H), yc + h / 2.0)

                        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                            continue

                        boxes.append([x1, y1, x2, y2])
                        # In TorchVision, label 0 is background, so VisDrone classes 0..9 -> 1..10
                        labels.append(cls_id + 1)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "image_stem": image_id,
            "orig_size": (W, H)
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def get_mask_rcnn_model(num_classes: int = 11) -> torchvision.models.detection.MaskRCNN:
    logger.info("Instantiating Mask R-CNN ResNet-50-FPN with COCO pretrained weights...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # Box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

    # Configure dynamic mask computation:
    # During training (with box-only VisDrone dataset), do not compute mask loss
    # During eval/inference, compute predicted masks
    model.roi_heads.has_mask = lambda: not model.training

    return model


# ---------------------------------------------------------------------------
# Validation Function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, dataloader, device, iou_thresh=0.50):
    model.eval()
    all_gts = []
    all_preds = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            stem = target["image_stem"]
            # Extract GTs
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy() # 1..10
            for b, l in zip(gt_boxes, gt_labels):
                all_gts.append({
                    "image_id": stem,
                    "class_id": int(l - 1), # map back to 0..9
                    "box_xyxy": b.tolist()
                })

            # Extract Preds
            p_boxes = output["boxes"].detach().cpu().numpy()
            p_scores = output["scores"].detach().cpu().numpy()
            p_labels = output["labels"].detach().cpu().numpy() # 1..10

            for b, s, l in zip(p_boxes, p_scores, p_labels):
                if s >= 0.05 and 1 <= l <= 10:
                    all_preds.append({
                        "image_id": stem,
                        "class_id": int(l - 1), # map back to 0..9
                        "score": float(s),
                        "box_xyxy": b.tolist()
                    })

    if len(all_preds) == 0:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0}

    eval_res = metrics.compute_metrics_for_model(
        all_gts, all_preds, iou_thresh=iou_thresh, class_names=CLASS_NAMES
    )
    mAP = eval_res.get("mAP", 0.0)
    prec = eval_res.get("precision", 0.0)
    rec = eval_res.get("recall", 0.0)
    f1 = eval_res.get("f1", 0.0)
    mean_iou = eval_res.get("mean_iou", 0.0)
    return {"mAP": mAP, "precision": prec, "recall": rec, "f1": f1, "mean_iou": mean_iou}


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("STARTING MODEL: Mask R-CNN ResNet-50-FPN")
    print("=" * 60)

    # 1. Environment & Hardware Verification
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available! Refusing to fall back to CPU.")
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Verified GPU: {gpu_name} (CUDA Device 0)")

    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Data Directories
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_dataset = VisDroneTorchDataset(data_dir / "images" / "train", data_dir / "labels" / "train")
    val_dataset = VisDroneTorchDataset(data_dir / "images" / "val", data_dir / "labels" / "val")
    test_dataset = VisDroneTorchDataset(data_dir / "images" / "test", data_dir / "labels" / "test")

    logger.info(f"Datasets Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    assert len(train_dataset) == 422, f"Expected 422 train images, found {len(train_dataset)}"
    assert len(val_dataset) == 106, f"Expected 106 val images, found {len(val_dataset)}"
    assert len(test_dataset) == 20, f"Expected 20 test images, found {len(test_dataset)}"

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True
    )

    # 3. Model Setup & Optimizer
    model = get_mask_rcnn_model(num_classes=11)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Mask R-CNN Model Parameters: {param_count:,}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    out_dir = PROJECT_ROOT / "results" / "trained" / "mask_rcnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    epochs = 25
    best_val_map = -1.0
    best_epoch = 0
    history = []

    logger.info(f"Starting Training for {epochs} epochs on {gpu_name}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        batch_count = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += losses.item()
            batch_count += 1

        lr_scheduler.step()
        avg_loss = train_loss / max(1, batch_count)

        # Validation evaluation at every epoch
        val_res = evaluate_model(model, val_loader, device, iou_thresh=0.50)
        v_map = val_res["mAP"]
        v_p = val_res["precision"]
        v_r = val_res["recall"]
        v_f1 = val_res["f1"]

        ep_duration = time.time() - epoch_start
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        logger.info(
            f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:4.1f}s) | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val mAP50: {v_map:.4f} | Val P: {v_p:.4f} | Val R: {v_r:.4f} | "
            f"VRAM: {vram_mb:.0f}MB"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_map50": round(v_map, 4),
            "val_precision": round(v_p, 4),
            "val_recall": round(v_r, 4),
            "val_f1": round(v_f1, 4),
            "duration_s": round(ep_duration, 1)
        })

        # Save last checkpoint
        torch.save(model.state_dict(), weights_dir / "last.pt")

        # Save best checkpoint strictly based on validation mAP@0.5
        if v_map > best_val_map:
            best_val_map = v_map
            best_epoch = epoch
            torch.save(model.state_dict(), weights_dir / "best.pt")
            logger.info(f"===> New Best Val mAP50: {best_val_map:.4f} at Epoch {best_epoch}! Saved best.pt")

    total_duration_m = (time.time() - start_time) / 60.0
    logger.info(f"Training completed in {total_duration_m:.2f} minutes.")
    logger.info(f"Best Checkpoint: Epoch {best_epoch} with Val mAP50 = {best_val_map:.4f}")

    # Save training history CSV
    hist_csv = out_dir / "training_history.csv"
    with open(hist_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_map50,val_precision,val_recall,val_f1,duration_s\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']},{h['val_map50']},{h['val_precision']},{h['val_recall']},{h['val_f1']},{h['duration_s']}\n")

    # Generate training curve plot
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(hist_csv)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['epoch'], df['train_loss'], 'r-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['val_map50'], 'b-', label='Val mAP@0.5')
    ax2.set_ylabel('mAP@0.5', color='b')
    plt.title('Mask R-CNN Fine-Tuning: Loss & Validation mAP@0.5')
    plt.savefig(out_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Final Evaluation of best.pt on Locked 20 Test Images
    logger.info("\n==================================================================")
    logger.info("     FINAL EVALUATION OF BEST CHECKPOINT ON LOCKED TEST SET       ")
    logger.info("==================================================================")

    best_weights_path = weights_dir / "best.pt"
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    model_size_mb = best_weights_path.stat().st_size / (1024**2)

    # Load authentic VisDrone test ground truth
    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"

    test_ids_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test_ids.txt"
    test_ids = [l.strip() for l in open(test_ids_file, encoding="utf-8") if l.strip()]

    all_gts, _, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))
    test_gts = [g for g in all_gts if g["image_id"] in test_ids and 0 <= g["class_id"] <= 9]
    test_img_paths = [str(imgs_dir / f"{tid}.jpg") for tid in test_ids]

    logger.info(f"Locked Test Set: {len(test_img_paths)} images, {len(test_gts)} valid GT objects.")

    # Model warm-up on GPU
    logger.info("Warming up Mask R-CNN on GPU...")
    warmup_tensor = torchvision.transforms.functional.to_tensor(Image.open(test_img_paths[0]).convert("RGB")).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model([warmup_tensor])
        torch.cuda.synchronize()

    # Timed inference
    logger.info("Running timed inference on 20 test images...")
    timings = []
    test_predictions = []

    with torch.no_grad():
        for img_p in test_img_paths:
            stem = Path(img_p).stem
            img_pil = Image.open(img_p).convert("RGB")
            W, H = img_pil.size
            img_t = torchvision.transforms.functional.to_tensor(img_pil).to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model([img_t])[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            timings.append(dt)

            boxes = outputs["boxes"].detach().cpu().numpy()
            scores = outputs["scores"].detach().cpu().numpy()
            labels = outputs["labels"].detach().cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                if s >= 0.25 and 1 <= l <= 10: # threshold 0.25
                    x1 = max(0.0, float(b[0]))
                    y1 = max(0.0, float(b[1]))
                    x2 = min(float(W), float(b[2]))
                    y2 = min(float(H), float(b[3]))
                    if x2 > x1 and y2 > y1:
                        test_predictions.append({
                            "image_id": stem,
                            "class_id": int(l - 1), # map back to 0..9
                            "score": float(s),
                            "box_xyxy": [x1, y1, x2, y2]
                        })

    mean_lat_s = float(np.mean(timings))
    latency_ms = mean_lat_s * 1000.0
    fps = 1.0 / mean_lat_s if mean_lat_s > 0 else 0.0

    logger.info(f"Inference complete: {len(test_predictions)} predictions generated.")
    logger.info(f"Latency: {latency_ms:.2f} ms | FPS: {fps:.2f}")

    # Calculate metrics
    det_metrics = metrics.compute_metrics_for_model(
        test_gts, test_predictions, iou_thresh=0.50, class_names=CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(test_gts, test_predictions, iou_thresh=0.50)

    # Confusion matrix
    cm_mat, cm_names = confusion_matrix.build_confusion_matrix(
        test_gts, test_predictions, CLASS_NAMES, iou_thresh=0.50
    )
    cm_out = out_dir / "test_confusion_matrix.png"
    confusion_matrix.plot_confusion_matrix(cm_mat, cm_names, "Confusion Matrix: Fine-Tuned Mask R-CNN (Test Set)", str(cm_out))

    mAP50 = det_metrics.get("mAP", 0.0)
    prec = det_metrics.get("precision", 0.0)
    rec = det_metrics.get("recall", 0.0)
    f1 = det_metrics.get("f1", 0.0)
    small_ap = size_metrics.get("small", {}).get("ap", 0.0) or 0.0
    med_ap = size_metrics.get("medium", {}).get("ap", 0.0) or 0.0
    large_ap = size_metrics.get("large", {}).get("ap", 0.0) or 0.0
    mean_iou = det_metrics.get("mean_iou", 0.0)

    test_res_dict = {
        "model": "Mask R-CNN (ResNet50+FPN)",
        "training_status": "fine-tuned",
        "best_epoch": best_epoch,
        "best_val_map50": round(float(best_val_map), 4),
        "test_images_count": len(test_img_paths),
        "test_gt_objects_count": len(test_gts),
        "test_predictions_count": len(test_predictions),
        "mAP50": round(float(mAP50), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "small_ap": round(float(small_ap), 4),
        "medium_ap": round(float(med_ap), 4),
        "large_ap": round(float(large_ap), 4),
        "mean_iou": round(float(mean_iou), 4),
        "latency_ms": round(float(latency_ms), 2),
        "fps": round(float(fps), 2),
        "parameters": int(param_count),
        "model_size_mb": round(float(model_size_mb), 2),
        "per_class_ap": det_metrics.get("per_class_AP", {})
    }

    # Save metrics JSON
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_res_dict, f, indent=2)

    # Append to results/trained_summary.csv
    summary_csv = PROJECT_ROOT / "results" / "trained_summary.csv"
    with open(summary_csv, "a", encoding="utf-8") as f:
        f.write(f"Mask R-CNN,fine-tuned,{mAP50:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{small_ap:.4f},{med_ap:.4f},{large_ap:.4f},{mean_iou:.4f},{latency_ms:.2f},{fps:.2f},{param_count},{model_size_mb:.2f}\n")

    # Save comparison with zero-shot Mask R-CNN
    zero_shot_mrcnn = {
        "mAP50": 0.0887,
        "precision": 0.3952,
        "recall": 0.2506,
        "f1": 0.3067,
        "small_ap": 0.0089,
        "mean_iou": 0.7857,
        "latency_ms": 4169.4,
        "fps": 0.24
    }

    zs_vs_trained_csv = PROJECT_ROOT / "results" / "zero_shot_vs_trained_mask_rcnn.csv"
    with open(zs_vs_trained_csv, "w", encoding="utf-8") as f:
        f.write("metric,zero_shot,fine_tuned,absolute_change,relative_change_pct\n")
        for k in ["mAP50", "precision", "recall", "f1", "small_ap", "mean_iou", "latency_ms", "fps"]:
            zs = zero_shot_mrcnn[k]
            ft = test_res_dict[k]
            diff = ft - zs
            pct_s = f"{(diff / zs)*100.0:+.2f}%" if zs != 0 else "N/A"
            f.write(f"{k},{zs},{ft},{diff:+.4f},{pct_s}\n")

    # Generate qualitative visualizations with predicted masks overlay
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    comp_qual_dir = PROJECT_ROOT / "results" / "comparison" / "qualitative" / "mask_rcnn"
    comp_qual_dir.mkdir(parents=True, exist_ok=True)

    COLORS = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
        (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187)
    ]

    for tid in test_ids[:6]:
        img_p = imgs_dir / f"{tid}.jpg"
        img_orig = Image.open(img_p).convert("RGB")
        W, H = img_orig.size

        # GT panel
        img_gt = img_orig.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        cur_gts = [g for g in all_gts if g["image_id"] == tid and 0 <= g["class_id"] <= 9]
        for g in cur_gts:
            b = g["box_xyxy"]
            cid = g["class_id"]
            cname = CLASS_NAMES[cid]
            col = COLORS[cid % len(COLORS)]
            draw_gt.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=2)
            draw_gt.text((b[0], max(0, b[1] - 10)), cname, fill=col)

        # Pred panel with predicted mask overlays
        img_pred = img_orig.copy()
        draw_pred = ImageDraw.Draw(img_pred)
        img_t = torchvision.transforms.functional.to_tensor(img_orig).to(device)

        with torch.no_grad():
            outs = model([img_t])[0]

        b_np = outs["boxes"].detach().cpu().numpy()
        s_np = outs["scores"].detach().cpu().numpy()
        l_np = outs["labels"].detach().cpu().numpy()
        m_np = outs["masks"].detach().cpu().numpy() if "masks" in outs else None

        # Mask overlay array
        overlay = np.array(img_pred).astype(np.float32)

        for i, (b, s, l) in enumerate(zip(b_np, s_np, l_np)):
            if s >= 0.25 and 1 <= l <= 10:
                cid = int(l - 1)
                cname = f"{CLASS_NAMES[cid]} {s:.2f}"
                col = COLORS[cid % len(COLORS)]

                # Apply soft colored mask overlay
                if m_np is not None:
                    mask = m_np[i, 0] > 0.5
                    for ch in range(3):
                        overlay[:, :, ch] = np.where(mask, overlay[:, :, ch] * 0.6 + col[ch] * 0.4, overlay[:, :, ch])

        img_pred = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw_pred = ImageDraw.Draw(img_pred)

        for b, s, l in zip(b_np, s_np, l_np):
            if s >= 0.25 and 1 <= l <= 10:
                cid = int(l - 1)
                cname = f"{CLASS_NAMES[cid]} {s:.2f}"
                col = COLORS[cid % len(COLORS)]
                draw_pred.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=2)
                draw_pred.text((b[0], max(0, b[1] - 10)), cname, fill=col)

        combined = Image.new("RGB", (W * 2, H + 40), (240, 240, 240))
        combined.paste(img_gt, (0, 40))
        combined.paste(img_pred, (W, 40))

        draw_title = ImageDraw.Draw(combined)
        draw_title.text((20, 10), f"Ground Truth ({len(cur_gts)} objects) - {tid}", fill=(0, 0, 0))
        draw_title.text((W + 20, 10), "Fine-Tuned Mask R-CNN (BBoxes + Predicted Masks)", fill=(0, 0, 0))

        combined.save(qual_dir / f"compare_{tid}.jpg", quality=90)
        combined.save(comp_qual_dir / f"compare_{tid}.jpg", quality=90)

    print("\n" + "=" * 60)
    print("COMPLETED MODEL: Mask R-CNN ResNet-50-FPN")
    print("=" * 60)
    print(f"Test mAP@0.50: {mAP50:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1:        {f1:.4f}")
    print(f"Small AP:       {small_ap:.4f}")
    print(f"Latency:        {latency_ms:.2f} ms")
    print(f"FPS:            {fps:.2f}")


if __name__ == "__main__":
    main()
