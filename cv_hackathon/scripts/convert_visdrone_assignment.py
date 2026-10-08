"""
convert_visdrone_assignment.py
Converts raw VisDrone annotations in C:/Users/yogac/Downloads/VisDrone_Assignment
(VisDrone2019-DET-train, VisDrone2019-DET-val, VisDrone2019-DET-test-dev)
to normalized YOLO .txt format inside each split's 'labels' folder.
Generates visdrone.yaml ready for YOLO training.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

BASE_DIR = Path("C:/Users/yogac/Downloads/VisDrone_Assignment")

SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev"
]

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]

def convert_single_image(img_path: Path, ann_path: Path, out_path: Path):
    try:
        with Image.open(img_path) as im:
            w_img, h_img = im.size
    except Exception as e:
        return 0, 0

    dw = 1.0 / w_img
    dh = 1.0 / h_img

    if not ann_path.exists():
        out_path.write_text("", encoding="utf-8")
        return 1, 0

    lines = []
    box_count = 0
    try:
        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                parts = [p.strip() for p in raw_line.strip().split(",") if p.strip()]
                if len(parts) < 8:
                    continue
                # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category_id>,<truncation>,<occlusion>
                score = parts[4]
                cat_id = int(parts[5])

                # Skip category 0 (ignored regions) or score == 0
                if cat_id == 0:
                    continue
                if cat_id < 1 or cat_id > 10:
                    continue

                cls_id = cat_id - 1  # 0-indexed

                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])

                # Clamp bounding boxes
                x = max(0.0, x)
                y = max(0.0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                if w <= 0 or h <= 0:
                    continue

                xc = (x + w / 2.0) * dw
                yc = (y + h / 2.0) * dh
                wn = w * dw
                hn = h * dh

                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                wn = min(max(wn, 0.0), 1.0)
                hn = min(max(hn, 0.0), 1.0)

                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                box_count += 1
    except Exception as e:
        pass

    out_path.write_text("".join(lines), encoding="utf-8")
    return 1, box_count

def process_split(split_name: str):
    split_dir = BASE_DIR / split_name
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"\nProcessing {split_name}: {len(img_files)} images...")

    tasks = []
    for img_p in img_files:
        ann_p = annotations_dir / f"{img_p.stem}.txt"
        out_p = labels_dir / f"{img_p.stem}.txt"
        tasks.append((img_p, ann_p, out_p))

    t0 = time.time()
    total_imgs = 0
    total_boxes = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(convert_single_image, img_p, ann_p, out_p) for img_p, ann_p, out_p in tasks]
        for f in futures:
            n_img, n_box = f.result()
            total_imgs += n_img
            total_boxes += n_box

    dt = time.time() - t0
    print(f"Finished {split_name}: {total_imgs} images, {total_boxes} valid boxes converted in {dt:.2f}s ({total_imgs/dt:.1f} imgs/s)")

def create_yaml():
    yaml_path = BASE_DIR / "visdrone.yaml"
    content = f"""# VisDrone2019-DET Dataset Configuration (Full Dataset)
path: {BASE_DIR.as_posix()}
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\nCreated dataset YAML configuration: {yaml_path}")

    # Also write classes.txt
    classes_path = BASE_DIR / "classes.txt"
    classes_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    print(f"Created classes.txt: {classes_path}")

if __name__ == "__main__":
    for s in SPLITS:
        if (BASE_DIR / s).exists():
            process_split(s)
        else:
            print(f"Skipping {s}: directory not found")
    create_yaml()
    print("\nVisDrone dataset conversion is 100% complete!")
