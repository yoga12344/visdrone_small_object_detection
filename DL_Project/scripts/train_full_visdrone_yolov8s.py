"""
train_full_visdrone_yolov8s.py
==============================
Training YOLOv8s on the complete VisDrone2019 dataset:
- Train: 6,471 images (343,204 targets)
- Validation: 548 images (38,759 targets)
- Test-Dev: 1,610 images (75,102 targets)
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
"""
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("STARTING FULL VISDRONE DATASET TRAINING (YOLOv8s)")
    print("=" * 70)

    # 1. Hardware Check
    assert torch.cuda.is_available(), "CUDA GPU is required but not available."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Compute Device : {gpu_name}")
    print(f"Total VRAM     : {vram_gb:.2f} GB")

    # 2. Dataset YAML Verification
    data_yaml = Path("C:/Users/yogac/Downloads/VisDrone_Assignment/visdrone.yaml").resolve()
    assert data_yaml.exists(), f"Dataset configuration not found at {data_yaml}"
    print(f"Dataset YAML   : {data_yaml}")

    # 3. Model Weights Initialization
    weights_path = Path("yolov8s.pt").resolve()
    assert weights_path.exists(), f"Pretrained weights not found at {weights_path}"
    print(f"Base Weights   : {weights_path}")

    # 4. Output Directories
    output_dir = Path("results/full_visdrone_training").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = YOLO(str(weights_path))

    # 5. Training Arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": 50,           # 50 epochs for high convergence
        "patience": 15,         # Early stopping if no improvement for 15 epochs
        "batch": 8,             # Optimal and stable for 8GB VRAM
        "imgsz": 640,           # Standard resolution
        "device": 0,            # CUDA:0
        "workers": 0,           # Required for Windows IPC stability
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "project": str(output_dir),
        "name": "yolov8s_full_6471imgs",
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": True
    }

    print("\nTraining Parameters:")
    for k, v in train_args.items():
        print(f"  - {k:15s}: {v}")

    print("\n" + "=" * 70)
    print("Launching training loop...")
    print("=" * 70 + "\n")

    t0 = time.time()
    results = model.train(**train_args)
    total_time = (time.time() - t0) / 60.0

    print("\n" + "=" * 70)
    print(f"Training completed in {total_time:.2f} minutes.")
    print(f"Artifacts saved in: {output_dir / 'yolov8s_full_6471imgs'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
