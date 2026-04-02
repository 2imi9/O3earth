#!/usr/bin/env python3
"""
evaluate.py — Comprehensive evaluation of fine-tuned OlmoEarth segmentation model.

Compares predicted output tiles against EIA ground-truth labels.
Reports per-class IoU, precision, recall, F1, and a confusion matrix.

Usage:
    # Evaluate val split (default)
    python scripts/evaluate.py --root ./dataset

    # Evaluate all windows (train + val)
    python scripts/evaluate.py --root ./dataset --split all

    # Save results to JSON
    python scripts/evaluate.py --root ./dataset --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sources.eia_860 import CLASS_NAMES, NUM_CLASSES


def load_geotiff(path):
    """Load a single-band GeoTIFF as numpy array."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            return src.read(1)
    except ImportError:
        # Fallback: try tifffile
        import tifffile

        return tifffile.imread(str(path))


def find_window_pairs(dataset_root, group="default", split=None):
    """
    Find paired (prediction, label) GeoTIFFs for each window.

    Returns list of (pred_path, label_path, window_name) tuples.
    """
    windows_dir = Path(dataset_root) / "windows" / group
    if not windows_dir.exists():
        print(f"Error: {windows_dir} does not exist")
        raise SystemExit(1)

    pairs = []
    for win_dir in sorted(windows_dir.iterdir()):
        if not win_dir.is_dir():
            continue

        # Filter by split tag if requested
        if split and split != "all":
            meta_path = win_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                win_split = meta.get("tags", {}).get("split", "")
                if win_split != split:
                    continue

        # Find output prediction layer
        pred_dir = win_dir / "layers" / "output"
        label_dir = win_dir / "layers" / "eia_energy"

        pred_files = list(pred_dir.glob("*/geotiff.tif")) if pred_dir.exists() else []
        label_files = (
            list(label_dir.glob("*/geotiff.tif")) if label_dir.exists() else []
        )

        if pred_files and label_files:
            pairs.append((pred_files[0], label_files[0], win_dir.name))

    return pairs


def compute_metrics(all_preds, all_labels, num_classes=NUM_CLASSES):
    """
    Compute per-class and aggregate metrics from flattened pred/label arrays.

    Returns dict with per-class IoU, precision, recall, F1, and confusion matrix.
    """
    # Confusion matrix: rows = true, cols = predicted
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            cm[true_cls, pred_cls] = np.sum(
                (all_labels == true_cls) & (all_preds == pred_cls)
            )

    metrics = {}
    per_class = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )

        per_class[c] = {
            "name": CLASS_NAMES.get(c, f"class_{c}"),
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[c, :].sum()),
        }

    # Mean IoU (excluding background class 0)
    valid_ious = [
        per_class[c]["iou"]
        for c in range(1, num_classes)
        if not np.isnan(per_class[c]["iou"]) and per_class[c]["support"] > 0
    ]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    # Overall accuracy
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0.0

    metrics["per_class"] = per_class
    metrics["mean_iou_no_bg"] = float(mean_iou)
    metrics["overall_accuracy"] = float(accuracy)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["total_pixels"] = int(total)

    return metrics


def print_report(metrics):
    """Pretty-print the evaluation report."""
    per_class = metrics["per_class"]

    print("\n" + "=" * 78)
    print("  OlmoEarth Energy Infrastructure Segmentation — Evaluation Report")
    print("=" * 78)

    # Per-class table
    header = f"{'Class':>2s}  {'Name':<12s}  {'IoU':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'Support':>9s}"
    print(f"\n{header}")
    print("-" * len(header))

    for c in range(NUM_CLASSES):
        info = per_class[c]
        iou_s = f"{info['iou']:.4f}" if not np.isnan(info["iou"]) else "   N/A"
        prec_s = (
            f"{info['precision']:.4f}" if not np.isnan(info["precision"]) else "   N/A"
        )
        rec_s = f"{info['recall']:.4f}" if not np.isnan(info["recall"]) else "   N/A"
        f1_s = f"{info['f1']:.4f}" if not np.isnan(info["f1"]) else "   N/A"
        print(
            f"  {c:2d}  {info['name']:<12s}  {iou_s:>7s}  {prec_s:>7s}  {rec_s:>7s}  {f1_s:>7s}  {info['support']:>9d}"
        )

    print("-" * len(header))
    print(
        f"  Mean IoU (excl. background): {metrics['mean_iou_no_bg']:.4f}"
    )
    print(f"  Overall pixel accuracy:      {metrics['overall_accuracy']:.4f}")
    print(f"  Total pixels evaluated:      {metrics['total_pixels']:,}")

    # Confusion matrix summary
    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    cm = np.array(metrics["confusion_matrix"])
    # Print abbreviated version (non-zero classes only)
    active = [c for c in range(NUM_CLASSES) if cm[c, :].sum() > 0 or cm[:, c].sum() > 0]
    if active:
        labels = [CLASS_NAMES.get(c, str(c))[:5] for c in active]
        print(f"  {'':>5s}  " + "  ".join(f"{l:>7s}" for l in labels))
        for r in active:
            row_label = CLASS_NAMES.get(r, str(r))[:5]
            vals = "  ".join(f"{cm[r, c]:>7d}" for c in active)
            print(f"  {row_label:>5s}  {vals}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OlmoEarth segmentation predictions"
    )
    parser.add_argument("--root", default="./dataset", help="rslearn dataset root")
    parser.add_argument("--group", default="default", help="Window group")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "all"],
        help="Which split to evaluate (default: val)",
    )
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Scanning for prediction/label pairs (split={args.split})...")
    pairs = find_window_pairs(args.root, group=args.group, split=args.split)
    print(f"  Found {len(pairs)} windows with both predictions and labels")

    if not pairs:
        print("\nNo prediction/label pairs found.")
        print("Make sure you've run:")
        print("  1. bash scripts/build_dataset.sh  (to materialize labels)")
        print("  2. bash scripts/train.sh          (to train the model)")
        print("  3. rslearn model predict --config configs/model.yaml")
        raise SystemExit(1)

    # Load all tiles
    print("Loading tiles...")
    all_preds = []
    all_labels = []
    for pred_path, label_path, name in pairs:
        pred = load_geotiff(pred_path).flatten()
        label = load_geotiff(label_path).flatten()
        # Ensure same length (crop to min)
        n = min(len(pred), len(label))
        all_preds.append(pred[:n])
        all_labels.append(label[:n])

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(f"  Total pixels: {len(all_preds):,}")

    # Compute and print metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics["num_windows"] = len(pairs)
    metrics["split"] = args.split
    print_report(metrics)

    # Save to JSON
    if args.output:
        # Convert NaN to None for JSON
        def clean_nan(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_nan(v) for v in obj]
            return obj

        out_path = Path(args.output)
        out_path.write_text(json.dumps(clean_nan(metrics), indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
