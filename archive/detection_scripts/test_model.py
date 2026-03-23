#!/usr/bin/env python3
"""
test_model.py — Quick test of the fine-tuned OlmoEarth energy segmentation model.

Loads the checkpoint directly, runs inference on validation windows, and reports:
  1. Per-class prediction distribution
  2. Per-class IoU / accuracy vs ground truth
  3. Saves sample prediction images for visual inspection

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --checkpoint project_data/openenergyengine/run_00/best.ckpt
    python scripts/test_model.py --max-windows 50 --save-images
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Class definitions
CLASS_NAMES = {
    0: "background",
    1: "solar",
    2: "wind",
    3: "gas",
    4: "coal",
    5: "nuclear",
    6: "hydro",
    7: "oil",
    8: "biomass",
    9: "geothermal",
    10: "storage",
}
NUM_CLASSES = 11


def load_checkpoint(ckpt_path, config_path="configs/model.yaml"):
    """Load the rslearn Lightning checkpoint using the training config."""
    import yaml
    from jsonargparse import Namespace
    from rslearn.train.lightning_module import RslearnLightningModule
    from rslearn.models.multitask import MultiTaskModel
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth
    from rslearn.models.unet import UNetDecoder
    from rslearn.train.tasks.segmentation import SegmentationHead, SegmentationTask
    from rslearn.train.tasks.multi_task import MultiTask
    from rslearn.train.optimizer import AdamW

    # Build model from config (same architecture as training)
    # Use the cached HF model path directly to avoid re-download issues
    cached_model_path = Path.home() / ".cache/huggingface/hub/models--allenai--OlmoEarth-v1-Base/snapshots"
    snapshot_dirs = list(cached_model_path.iterdir()) if cached_model_path.exists() else []
    if snapshot_dirs:
        model_path = str(snapshot_dirs[0])
        print(f"  Using cached OlmoEarth from: {model_path}")
        encoder = OlmoEarth(model_id=model_path, patch_size=8, selector=["encoder"], random_initialization=True)
    else:
        encoder = OlmoEarth(model_id="OLMOEARTH_V1_BASE", patch_size=8, selector=["encoder"], random_initialization=True)
    decoder = UNetDecoder(in_channels=[[8, 768]], out_channels=11)
    seg_head = SegmentationHead(
        weights=[0.02, 5.0, 8.0, 6.0, 15.0, 20.0, 8.0, 10.0, 12.0, 18.0, 10.0],
        dice_loss=True,
    )

    multi_model = MultiTaskModel(
        encoder=[encoder],
        decoders={"energy_segmentation": [decoder, seg_head]},
    )

    seg_task = SegmentationTask(
        num_classes=11,
        enable_miou_metric=True,
        zero_is_invalid=False,
        nodata_value=255,
    )
    task = MultiTask(
        tasks={"energy_segmentation": seg_task},
        input_mapping={"energy_segmentation": {"label": "targets"}},
    )

    optimizer = AdamW(lr=0.0001)

    lightning_module = RslearnLightningModule(
        model=multi_model,
        task=task,
        optimizer=optimizer,
    )

    # Load weights from checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    lightning_module.load_state_dict(ckpt["state_dict"])
    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    print(f"  Model loaded on: {device}")
    print(f"  Parameters: {sum(p.numel() for p in lightning_module.parameters()):,}")
    return lightning_module, device


def load_window_data(win_dir):
    """Load sentinel2 imagery and label for a window."""
    import tifffile

    # Load sentinel2 bands
    s2_dir = win_dir / "layers" / "sentinel2_l2a"
    s2_subdirs = sorted([d for d in s2_dir.iterdir() if d.is_dir()]) if s2_dir.exists() else []
    if not s2_subdirs:
        return None, None

    s2_tif = s2_subdirs[0] / "geotiff.tif"
    if not s2_tif.exists():
        return None, None

    imagery = tifffile.imread(str(s2_tif)).astype(np.float32)  # (C, H, W) or (H, W, C)
    if imagery.ndim == 3 and imagery.shape[0] not in [12, 13]:
        imagery = imagery.transpose(2, 0, 1)  # HWC -> CHW

    # Load label
    label_tif = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
    if label_tif.exists():
        label = tifffile.imread(str(label_tif)).astype(np.int64)
        if label.ndim == 3:
            label = label[0]
    else:
        label = None

    return imagery, label


def normalize_sentinel2(imagery):
    """Apply OlmoEarth normalization to Sentinel-2 bands."""
    # OlmoEarth expects bands in specific order and normalized
    # The bands in our data: B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09
    # Simple normalization: divide by 10000 (Sentinel-2 reflectance scale)
    imagery = imagery / 10000.0
    imagery = np.clip(imagery, 0, 1)
    return imagery


def run_inference(model, device, imagery):
    """Run model inference on a single window."""
    from rslearn.models.multitask import ModelContext
    from rslearn.models.olmoearth_pretrain.model import RasterImage

    # Normalize
    imagery = normalize_sentinel2(imagery)

    # To tensor: (C, H, W) -> RasterImage expects 4D CTHW
    x = torch.from_numpy(imagery).float().to(device)
    # Add time dimension: (C, H, W) -> (C, 1, H, W)
    x = x.unsqueeze(1)
    raster = RasterImage(image=x)

    # Create ModelContext as expected by rslearn
    context = ModelContext(
        inputs=[{"sentinel2_l2a": raster}],
        metadatas=[None],
    )

    with torch.no_grad():
        output = model.model(context)

    # Unwrap ModelOutput -> extract the actual logits tensor
    def extract_logits(obj, depth=0):
        """Recursively unwrap until we find a tensor."""
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'outputs'):
            return extract_logits(obj.outputs, depth + 1)
        if isinstance(obj, dict):
            # Prefer energy_segmentation key
            if "energy_segmentation" in obj:
                return extract_logits(obj["energy_segmentation"], depth + 1)
            return extract_logits(next(iter(obj.values())), depth + 1)
        if isinstance(obj, (list, tuple)):
            return extract_logits(obj[0], depth + 1)
        raise ValueError(f"Cannot extract logits from {type(obj)} at depth {depth}")

    logits = extract_logits(output)

    # logits shape: (batch, num_classes, H, W) or (num_classes, H, W)
    if logits.dim() == 4:
        pred = logits[0].argmax(dim=0).cpu().numpy()
    elif logits.dim() == 3:
        pred = logits.argmax(dim=0).cpu().numpy()
    else:
        pred = logits.cpu().numpy()
    return pred


def compute_metrics(all_preds, all_labels):
    """Compute per-class IoU and overall accuracy."""
    results = {}
    for c in range(NUM_CLASSES):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        results[c] = {
            "name": CLASS_NAMES[c],
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "support": int(np.sum(all_labels == c)),
            "predicted": int(np.sum(all_preds == c)),
        }

    total = len(all_preds)
    correct = np.sum(all_preds == all_labels)
    results["accuracy"] = float(correct / total) if total > 0 else 0.0
    results["total_pixels"] = int(total)

    # Mean IoU (excluding background)
    valid_ious = [
        results[c]["iou"]
        for c in range(1, NUM_CLASSES)
        if not np.isnan(results[c]["iou"]) and results[c]["support"] > 0
    ]
    results["mean_iou"] = float(np.mean(valid_ious)) if valid_ious else 0.0

    return results


def save_prediction_image(pred, label, out_path):
    """Save a color-coded prediction vs label comparison image."""
    try:
        from PIL import Image
    except ImportError:
        return

    # Color palette for classes
    colors = [
        (0, 0, 0),        # 0 background - black
        (255, 255, 0),    # 1 solar - yellow
        (0, 200, 255),    # 2 wind - cyan
        (255, 128, 0),    # 3 gas - orange
        (128, 128, 128),  # 4 coal - gray
        (255, 0, 255),    # 5 nuclear - magenta
        (0, 0, 255),      # 6 hydro - blue
        (139, 69, 19),    # 7 oil - brown
        (0, 128, 0),      # 8 biomass - green
        (255, 0, 0),      # 9 geothermal - red
        (0, 255, 128),    # 10 storage - spring green
    ]

    def class_to_rgb(arr):
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c, color in enumerate(colors):
            mask = arr == c
            rgb[mask] = color
        return rgb

    pred_rgb = class_to_rgb(pred)
    if label is not None:
        label_rgb = class_to_rgb(label)
        # Side by side: label | prediction
        combined = np.concatenate([label_rgb, pred_rgb], axis=1)
    else:
        combined = pred_rgb

    Image.fromarray(combined).save(str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned OlmoEarth model")
    parser.add_argument(
        "--checkpoint",
        default="project_data/openenergyengine/run_02/best.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--root", default="./dataset", help="Dataset root")
    parser.add_argument("--group", default="ready", help="Window group")
    parser.add_argument("--split", default="val", help="Split to test on")
    parser.add_argument("--max-windows", type=int, default=20, help="Max windows to test")
    parser.add_argument("--save-images", action="store_true", help="Save prediction images")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for images")
    args = parser.parse_args()

    print("=" * 70)
    print("  OlmoEarth Energy Segmentation — Model Test")
    print("=" * 70)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, device = load_checkpoint(args.checkpoint)

    # Find val windows
    windows_dir = Path(args.root) / "windows" / args.group
    val_windows = []
    for win_dir in sorted(windows_dir.iterdir()):
        if not win_dir.is_dir():
            continue
        meta_path = win_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("tags", {}).get("split") == args.split:
                val_windows.append(win_dir)

    print(f"\nFound {len(val_windows)} {args.split} windows")
    if not val_windows:
        print("No windows found! Check --group and --split flags.")
        return

    test_windows = val_windows[: args.max_windows]
    print(f"Testing on {len(test_windows)} windows...\n")

    if args.save_images:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)

    all_preds = []
    all_labels = []
    class_windows = {c: 0 for c in range(NUM_CLASSES)}  # windows containing each class
    skipped = 0

    for i, win_dir in enumerate(test_windows):
        imagery, label = load_window_data(win_dir)
        if imagery is None:
            skipped += 1
            continue

        pred = run_inference(model, device, imagery)
        if pred is None:
            skipped += 1
            continue

        # Track which classes appear
        unique_pred = set(np.unique(pred))
        for c in unique_pred:
            class_windows[c] += 1

        if label is not None:
            n = min(pred.size, label.size)
            all_preds.append(pred.flatten()[:n])
            all_labels.append(label.flatten()[:n])

        if args.save_images:
            save_prediction_image(pred, label, out_dir / f"{win_dir.name}.png")

        # Progress
        if (i + 1) % 5 == 0 or i == len(test_windows) - 1:
            print(f"  Processed {i + 1}/{len(test_windows)} windows")

    print(f"\n  Skipped {skipped} windows (missing data)")

    # Prediction distribution
    print(f"\n{'-' * 50}")
    print("  Prediction Distribution (windows containing class):")
    print(f"{'-' * 50}")
    for c in range(NUM_CLASSES):
        bar = "#" * class_windows[c]
        print(f"  {c:2d} {CLASS_NAMES[c]:<12s}  {class_windows[c]:>3d} windows  {bar}")

    # Metrics vs ground truth
    if all_preds:
        all_preds_flat = np.concatenate(all_preds)
        all_labels_flat = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds_flat, all_labels_flat)

        print(f"\n{'-' * 70}")
        print("  Metrics vs Ground Truth Labels")
        print(f"{'-' * 70}")
        print(f"  {'Class':>2s}  {'Name':<12s}  {'IoU':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'Support':>9s}  {'Predicted':>9s}")
        print(f"  {'-' * 64}")

        for c in range(NUM_CLASSES):
            m = metrics[c]
            iou_s = f"{m['iou']:.4f}" if not np.isnan(m["iou"]) else "   N/A"
            prec_s = f"{m['precision']:.4f}" if not np.isnan(m["precision"]) else "   N/A"
            rec_s = f"{m['recall']:.4f}" if not np.isnan(m["recall"]) else "   N/A"
            print(f"  {c:2d}  {m['name']:<12s}  {iou_s:>7s}  {prec_s:>7s}  {rec_s:>7s}  {m['support']:>9d}  {m['predicted']:>9d}")

        print(f"  {'-' * 64}")
        print(f"  Mean IoU (excl. bg): {metrics['mean_iou']:.4f}")
        print(f"  Overall accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Total pixels:        {metrics['total_pixels']:,}")

    if args.save_images:
        print(f"\n  Prediction images saved to: {args.output_dir}/")
        print(f"  (Left = ground truth, Right = prediction)")

    print(f"\n{'=' * 70}")
    print("  Test complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
