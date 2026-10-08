"""
evaluator.py — High-level evaluation runner for CV hackathon experiments.

Wraps :mod:`metrics` to support multi-model comparison, formatted console
summaries, and JSON persistence.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

try:
    from .metrics import compute_metrics_for_model, compute_size_metrics
except ImportError:
    from metrics import compute_metrics_for_model, compute_size_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates evaluation of one or more object-detection models against a
    fixed ground-truth annotation set.

    Parameters
    ----------
    gt_annotations : list of dict
        Ground-truth entries; each must contain:
        ``{image_id: str|int, class_id: int, box_xyxy: [x1,y1,x2,y2]}``.
    class_names : dict
        ``{class_id: name}`` mapping used for human-readable output.
    iou_thresh : float
        IoU threshold for TP/FP assignment (default 0.5).
    device : str
        Stored for reference (e.g. ``'cpu'``, ``'cuda'``). The evaluator
        itself is device-agnostic (pure NumPy).

    Example
    -------
    >>> ev = Evaluator(gt_annotations, class_names={0: "car", 1: "person"})
    >>> ev.add_predictions("YOLOv8", yolo_preds)
    >>> ev.add_predictions("Faster-RCNN", rcnn_preds)
    >>> results = ev.evaluate_all()
    >>> ev.print_summary_table(results)
    >>> ev.save_results(results, "results/comparison.json")
    """

    def __init__(
        self,
        gt_annotations: List[Dict],
        class_names: Dict[int, str],
        iou_thresh: float = 0.5,
        device: str = "cpu",
    ) -> None:
        if not gt_annotations:
            raise ValueError("gt_annotations must not be empty.")
        self.gt_annotations = gt_annotations
        self.class_names = class_names
        self.iou_thresh = iou_thresh
        self.device = device
        self._predictions: Dict[str, List[Dict]] = {}  # model_name -> pred list
        logger.info(
            "Evaluator initialised | GT boxes=%d | classes=%d | IoU=%.2f | device=%s",
            len(gt_annotations),
            len(class_names),
            iou_thresh,
            device,
        )

    # ------------------------------------------------------------------
    # Prediction registration
    # ------------------------------------------------------------------

    def add_predictions(self, model_name: str, pred_annotations: List[Dict]) -> None:
        """
        Register a model's predictions for later evaluation.

        Parameters
        ----------
        model_name : str
            Human-readable model identifier (used as table column header).
        pred_annotations : list of dict
            Each entry: ``{image_id, class_id, score, box_xyxy}``.
        """
        if not pred_annotations:
            logger.warning("Model '%s' has 0 predictions — registering anyway.", model_name)
        self._predictions[model_name] = pred_annotations
        logger.info(
            "Registered %d predictions for model '%s'.",
            len(pred_annotations),
            model_name,
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self) -> Dict[str, Dict]:
        """
        Evaluate every registered model against the ground truth.

        Returns
        -------
        dict
            ``{model_name: metrics_dict}`` where ``metrics_dict`` is the
            output of :func:`~metrics.compute_metrics_for_model` extended
            with size-stratified metrics under key ``'size_metrics'``.
        """
        if not self._predictions:
            raise RuntimeError(
                "No predictions have been added. Call add_predictions() first."
            )

        results: Dict[str, Dict] = {}
        for model_name, preds in self._predictions.items():
            logger.info("Evaluating model: %s", model_name)
            try:
                metrics = compute_metrics_for_model(
                    self.gt_annotations,
                    preds,
                    iou_thresh=self.iou_thresh,
                    class_names=self.class_names,
                )
                size_metrics = compute_size_metrics(
                    self.gt_annotations,
                    preds,
                    iou_thresh=self.iou_thresh,
                )
                metrics["size_metrics"] = size_metrics
                results[model_name] = metrics
            except Exception as exc:
                logger.error(
                    "Evaluation of model '%s' failed: %s", model_name, exc, exc_info=True
                )
                results[model_name] = {"error": str(exc)}

        return results

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_summary_table(self, results: Dict[str, Dict]) -> None:
        """
        Print a formatted comparison table to stdout.

        Columns: model, mAP, precision, recall, F1, mean_IoU, TP, FP, FN.
        """
        models = [m for m, v in results.items() if "error" not in v]
        if not models:
            print("[Evaluator] No valid results to display.")
            return

        # Column definitions: (header, key, width, fmt)
        COLS = [
            ("Model",     None,         22, "s"),
            ("mAP",       "mAP",        8,  ".4f"),
            ("Precision", "precision",  10, ".4f"),
            ("Recall",    "recall",     9,  ".4f"),
            ("F1",        "f1",         8,  ".4f"),
            ("mIoU",      "mean_iou",   8,  ".4f"),
            ("TP",        "n_tp",       7,  "d"),
            ("FP",        "n_fp",       7,  "d"),
            ("FN",        "n_fn",       7,  "d"),
        ]

        sep = "+" + "+".join("-" * (w + 2) for _, _, w, _ in COLS) + "+"
        header_cells = []
        for h, _, w, _ in COLS:
            header_cells.append(f" {h:<{w}} ")
        header = "|" + "|".join(header_cells) + "|"

        print("\n" + sep)
        print(header)
        print(sep)

        for model in models:
            m = results[model]
            cells = []
            for h, key, w, fmt in COLS:
                if key is None:
                    cells.append(f" {model:<{w}} ")
                else:
                    val = m.get(key, "N/A")
                    if isinstance(val, float):
                        cells.append(f" {val:{w}.{fmt[-1] if fmt != 'd' else ''}}")
                    elif isinstance(val, int):
                        cells.append(f" {val:>{w}d} ")
                    else:
                        cells.append(f" {str(val):<{w}} ")
            print("|" + "|".join(cells) + "|")

        print(sep)

        # Per-class AP block
        print("\n--- Per-Class AP ---")
        all_class_names = sorted(
            {name for m in models for name in results[m].get("per_class_AP", {}).keys()}
        )
        col_w = max((len(n) for n in all_class_names), default=10) + 2
        header2 = f"{'Class':<{col_w}}" + "".join(f"{mn:>10}" for mn in models)
        print(header2)
        print("-" * len(header2))
        for cls_name in all_class_names:
            row = f"{cls_name:<{col_w}}"
            for mn in models:
                ap = results[mn].get("per_class_AP", {}).get(cls_name, "N/A")
                row += f"  {ap:>6.4f}  " if isinstance(ap, float) else f"  {'N/A':>6}  "
            print(row)

        # Size-stratified block
        print("\n--- Size-Stratified AP (small / medium / large) ---")
        size_header = f"{'Model':<22}{'Small AP':>12}{'Medium AP':>12}{'Large AP':>12}"
        print(size_header)
        print("-" * len(size_header))
        for mn in models:
            sm = results[mn].get("size_metrics", {})
            def _fmt(s_dict: Dict) -> str:
                if s_dict is None:
                    return "  N/A"
                if s_dict.get("ap") is None:
                    return "  N/A"
                return f"{s_dict['ap']:>10.4f}"
            print(
                f"{mn:<22}"
                f"{_fmt(sm.get('small', {})):>12}"
                f"{_fmt(sm.get('medium', {})):>12}"
                f"{_fmt(sm.get('large', {})):>12}"
            )
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, Dict], output_path: str) -> None:
        """
        Serialise evaluation results to a JSON file.

        Precision/recall curves (large numpy arrays) are stored as lists.
        The file is written atomically via a temporary sibling file.

        Parameters
        ----------
        results : dict
            Output of :meth:`evaluate_all`.
        output_path : str
            Destination file path (created if necessary, parent dirs included).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _make_serialisable(obj):
            """Recursively convert numpy scalars / arrays to Python types."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: _make_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serialisable(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "iou_thresh": self.iou_thresh,
            "n_gt_total": len(self.gt_annotations),
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "results": _make_serialisable(results),
        }

        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, output_path)
        logger.info("Results saved to %s", output_path)
        print(f"[Evaluator] Results saved → {output_path}")

    # ------------------------------------------------------------------
    # Experiment header
    # ------------------------------------------------------------------

    def print_experiment_header(self, config: Dict) -> None:
        """
        Print a mandatory experiment summary block to stdout.

        Parameters
        ----------
        config : dict
            Expected keys (all optional but recommended):
            ``dataset_name``, ``n_train``, ``n_val``, ``n_test``,
            ``class_names``, ``gt_source``, ``iou_thresh``,
            ``model_names``, ``notes``.
        """
        width = 70
        line  = "=" * width

        def _row(label: str, value) -> str:
            return f"  {label:<28}: {value}"

        print(f"\n{line}")
        print(f"{'EXPERIMENT SUMMARY':^{width}}")
        print(line)
        print(_row("Timestamp (UTC)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        print(_row("Dataset", config.get("dataset_name", "NOT SPECIFIED")))
        print(_row("Train images", config.get("n_train", "N/A")))
        print(_row("Val images",   config.get("n_val",   "N/A")))
        print(_row("Test images",  config.get("n_test",  "N/A")))
        print(_row("GT total boxes", len(self.gt_annotations)))

        cls_display = config.get(
            "class_names",
            list(self.class_names.values()) if self.class_names else "N/A",
        )
        if isinstance(cls_display, (list, tuple)):
            cls_display = ", ".join(str(c) for c in cls_display)
        print(_row("Classes", cls_display))

        print(_row("GT source / format", config.get("gt_source", "NOT SPECIFIED")))
        print(_row("IoU threshold", f"{self.iou_thresh:.2f}"))
        print(_row("Device", self.device))

        models = config.get("model_names", list(self._predictions.keys()))
        print(_row("Models to evaluate", ", ".join(models) if models else "None registered"))

        notes = config.get("notes", "")
        if notes:
            print(_row("Notes", notes))

        print(line + "\n")
