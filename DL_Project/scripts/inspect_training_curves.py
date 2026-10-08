import pandas as pd
from pathlib import Path

csv_path = Path("results/final_optimization/training/yolov8s_100ep_640/results.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print("Columns:", df.columns.tolist())
    print("Total epochs trained:", len(df))
    cols = ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    print("\nFirst 3 epochs:")
    print(df[cols].head(3).to_string(index=False))
    print("\nLast 5 epochs:")
    print(df[cols].tail(5).to_string(index=False))
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_row = df.loc[best_idx]
    print(f"\nBest val mAP50: {best_row['metrics/mAP50(B)']:.4f} at epoch {int(best_row['epoch'])}")
    print(f"Val box loss at best epoch: {best_row['val/box_loss']:.4f}")
    print(f"Final val box loss: {df['val/box_loss'].iloc[-1]:.4f}")
    print(f"Min val box loss: {df['val/box_loss'].min():.4f} at epoch {int(df.loc[df['val/box_loss'].idxmin()]['epoch'])}")
else:
    print(f"File not found: {csv_path}")
