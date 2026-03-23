#!/usr/bin/env python3
"""
test_model_v3.py — Test with post-processing: threshold tuning + size filtering.
Zero extra compute cost — just smarter predictions from existing model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

CLASS_NAMES = {0: "background", 1: "solar", 2: "wind"}
NUM_CLASSES = 3

BAND_ORDER = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
BAND_STATS = {
    "B01": {"mean": 1115.85, "std": 1955.70},
    "B02": {"mean": 1188.94, "std": 1859.19},
    "B03": {"mean": 1407.77, "std": 1727.74},
    "B04": {"mean": 1513.06, "std": 1740.78},
    "B05": {"mean": 1890.99, "std": 1754.73},
    "B06": {"mean": 2483.78, "std": 1622.12},
    "B07": {"mean": 2722.73, "std": 1621.82},
    "B08": {"mean": 2755.48, "std": 1612.26},
    "B09": {"mean": 3269.81, "std": 2651.09},
    "B11": {"mean": 2562.85, "std": 1441.55},
    "B12": {"mean": 1914.14, "std": 1328.89},
    "B8A": {"mean": 2885.57, "std": 1611.36},
}
STD_MULTIPLIER = 2


def normalize_olmoearth(imagery):
    result = np.zeros_like(imagery, dtype=np.float32)
    for i, band_name in enumerate(BAND_ORDER):
        stats = BAND_STATS[band_name]
        min_val = stats["mean"] - STD_MULTIPLIER * stats["std"]
        max_val = stats["mean"] + STD_MULTIPLIER * stats["std"]
        result[i] = (imagery[i] - min_val) / (max_val - min_val)
    return result


def load_checkpoint(ckpt_path):
    from rslearn.models.multitask import MultiTaskModel
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth
    from rslearn.models.unet import UNetDecoder
    from rslearn.train.tasks.segmentation import SegmentationHead, SegmentationTask
    from rslearn.train.tasks.multi_task import MultiTask
    from rslearn.train.optimizer import AdamW
    from rslearn.train.lightning_module import RslearnLightningModule

    cached_model_path = Path.home() / ".cache/huggingface/hub/models--allenai--OlmoEarth-v1-Base/snapshots"
    snapshot_dirs = list(cached_model_path.iterdir()) if cached_model_path.exists() else []
    model_path = str(snapshot_dirs[0]) if snapshot_dirs else "OLMOEARTH_V1_BASE"

    encoder = OlmoEarth(model_id=model_path, patch_size=4, selector=["encoder"], random_initialization=True)
    decoder = UNetDecoder(
        in_channels=[[4, 768]], out_channels=3,
        conv_layers_per_resolution=2, kernel_size=3,
        num_channels={4: 512, 2: 256, 1: 128},
    )
    seg_head = SegmentationHead(weights=[0.1, 5.0, 8.0], dice_loss=True)
    multi_model = MultiTaskModel(
        encoder=[encoder],
        decoders={"energy_segmentation": [decoder, seg_head]},
    )
    seg_task = SegmentationTask(num_classes=3, enable_miou_metric=True, zero_is_invalid=False, nodata_value=255)
    task = MultiTask(tasks={"energy_segmentation": seg_task}, input_mapping={"energy_segmentation": {"label": "targets"}})
    optimizer = AdamW(lr=0.0001)

    lightning_module = RslearnLightningModule(model=multi_model, task=task, optimizer=optimizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    lightning_module.load_state_dict(ckpt["state_dict"])
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    return lightning_module, device


def load_window_data(win_dir):
    import tifffile
    s2_dir = win_dir / "layers" / "sentinel2_l2a"
    s2_subdirs = sorted([d for d in s2_dir.iterdir() if d.is_dir()]) if s2_dir.exists() else []
    if not s2_subdirs:
        return None, None

    s2_tif = s2_subdirs[0] / "geotiff.tif"
    if not s2_tif.exists():
        return None, None

    imagery = tifffile.imread(str(s2_tif)).astype(np.float32)
    if imagery.ndim == 3 and imagery.shape[0] not in [12, 13]:
        imagery = imagery.transpose(2, 0, 1)

    label_tif = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
    if label_tif.exists():
        label = tifffile.imread(str(label_tif)).astype(np.int64)
        if label.ndim == 3:
            label = label[0]
    else:
        label = None
    return imagery, label


def run_inference_raw(model, device, imagery):
    """Returns raw logits instead of argmax predictions."""
    from rslearn.models.multitask import ModelContext
    from rslearn.models.olmoearth_pretrain.model import RasterImage

    imagery = normalize_olmoearth(imagery)
    x = torch.from_numpy(imagery).float().to(device)
    x = x.unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)
    raster = RasterImage(image=x)
    context = ModelContext(inputs=[{"sentinel2_l2a": raster}], metadatas=[None])

    with torch.no_grad():
        output = model.model(context)

    def extract_logits(obj, depth=0):
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'outputs'):
            return extract_logits(obj.outputs, depth + 1)
        if isinstance(obj, dict):
            if "energy_segmentation" in obj:
                return extract_logits(obj["energy_segmentation"], depth + 1)
            return extract_logits(next(iter(obj.values())), depth + 1)
        if isinstance(obj, (list, tuple)):
            return extract_logits(obj[0], depth + 1)
        raise ValueError(f"Cannot extract logits from {type(obj)} at depth {depth}")

    logits = extract_logits(output)

    if logits.dim() == 4:
        logits = logits[0]  # (C, H, W)

    return logits.cpu()


def postprocess(logits, conf_threshold=0.5, min_pixels=16):
    """
    Post-processing to boost precision:
    1. Softmax to get probabilities
    2. Only predict non-bg if confidence > threshold
    3. Remove small connected components (< min_pixels)
    """
    probs = torch.softmax(logits, dim=0).numpy()  # (C, H, W)
    pred = np.argmax(probs, axis=0)  # (H, W)

    # Threshold: only keep non-bg predictions where confidence exceeds threshold
    for c in range(1, NUM_CLASSES):
        mask = pred == c
        low_conf = probs[c] < conf_threshold
        pred[mask & low_conf] = 0  # revert to background

    # Size filter: remove small connected components for each non-bg class
    for c in range(1, NUM_CLASSES):
        mask = (pred == c).astype(np.int32)
        if mask.sum() == 0:
            continue
        labeled, n_features = ndimage.label(mask)
        for region_id in range(1, n_features + 1):
            region = labeled == region_id
            if region.sum() < min_pixels:
                pred[region] = 0  # too small, likely noise

    return pred, probs


def evaluate(all_preds, all_labels, label=""):
    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)

    print(f"\n{'=' * 70}")
    print(f"  Results: {label}")
    print(f"{'=' * 70}")
    print(f"  {'C':>2s}  {'Name':<12s}  {'IoU':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'Support':>9s}  {'Predicted':>9s}")
    print(f"  {'-' * 64}")

    valid_ious = []
    for c in range(NUM_CLASSES):
        tp = int(np.sum((all_preds_flat == c) & (all_labels_flat == c)))
        fp = int(np.sum((all_preds_flat == c) & (all_labels_flat != c)))
        fn = int(np.sum((all_preds_flat != c) & (all_labels_flat == c)))
        support = int(np.sum(all_labels_flat == c))
        predicted = int(np.sum(all_preds_flat == c))

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        iou_s = f"{iou:.4f}" if not np.isnan(iou) else "   N/A"
        prec_s = f"{prec:.4f}" if not np.isnan(prec) else "   N/A"
        rec_s = f"{rec:.4f}" if not np.isnan(rec) else "   N/A"
        print(f"  {c:2d}  {CLASS_NAMES[c]:<12s}  {iou_s:>7s}  {prec_s:>7s}  {rec_s:>7s}  {support:>9d}  {predicted:>9d}")

        if c > 0 and not np.isnan(iou) and support > 0:
            valid_ious.append(iou)

    total = len(all_preds_flat)
    correct = int(np.sum(all_preds_flat == all_labels_flat))
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    print(f"  {'-' * 64}")
    print(f"  Mean IoU (excl. bg): {miou:.4f}")
    print(f"  Overall accuracy:    {correct / total:.4f}")
    print(f"  Total pixels:        {total:,}")
    return miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="project_data/openenergyengine/run_15_3class_v2/best.ckpt")
    parser.add_argument("--root", default="./dataset")
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", default="test_results_v3")
    args = parser.parse_args()

    print("=" * 70)
    print("  OlmoEarth Test v3 — Post-Processing Sweep")
    print("  Testing: threshold tuning + size filtering")
    print("=" * 70)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, device = load_checkpoint(args.checkpoint)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Find val windows WITH labels only
    windows_dir = Path(args.root) / "windows" / "ready"
    val_windows = []
    for win_dir in sorted(windows_dir.iterdir()):
        if not win_dir.is_dir():
            continue
        meta_path = win_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("options", {}).get("split") != "val":
            continue
        label_tif = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_tif.exists():
            continue
        val_windows.append(win_dir)

    print(f"\nFound {len(val_windows)} val windows with labels")
    test_windows = val_windows[:args.max_windows]
    print(f"Testing on {len(test_windows)} windows...")

    if args.save_images:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)

    # Collect raw logits for all windows
    all_logits = []
    all_labels = []
    skipped = 0

    for i, win_dir in enumerate(test_windows):
        imagery, label = load_window_data(win_dir)
        if imagery is None:
            skipped += 1
            continue

        logits = run_inference_raw(model, device, imagery)
        all_logits.append((logits, win_dir.name))
        if label is not None:
            all_labels.append(label)
        else:
            all_labels.append(np.zeros(logits.shape[1:], dtype=np.int64))

        if (i + 1) % 20 == 0 or i == len(test_windows) - 1:
            print(f"  Processed {i + 1}/{len(test_windows)} (skipped {skipped})")

    # === SWEEP: try different thresholds and min sizes ===
    print(f"\n{'#' * 70}")
    print(f"  POST-PROCESSING SWEEP")
    print(f"{'#' * 70}")

    best_miou = 0
    best_config = None

    configs = [
        # (threshold, min_pixels, label)
        (0.0, 0, "Raw argmax (no post-processing)"),
        (0.3, 0, "Threshold=0.3, no size filter"),
        (0.4, 0, "Threshold=0.4, no size filter"),
        (0.5, 0, "Threshold=0.5, no size filter"),
        (0.6, 0, "Threshold=0.6, no size filter"),
        (0.7, 0, "Threshold=0.7, no size filter"),
        (0.3, 16, "Threshold=0.3, min_px=16"),
        (0.4, 16, "Threshold=0.4, min_px=16"),
        (0.5, 16, "Threshold=0.5, min_px=16"),
        (0.5, 32, "Threshold=0.5, min_px=32"),
        (0.5, 64, "Threshold=0.5, min_px=64"),
        (0.4, 32, "Threshold=0.4, min_px=32"),
        (0.6, 32, "Threshold=0.6, min_px=32"),
    ]

    for conf_thresh, min_px, label in configs:
        preds_list = []
        labels_list = []

        for (logits, name), label_arr in zip(all_logits, all_labels):
            if conf_thresh == 0.0 and min_px == 0:
                pred = logits.argmax(dim=0).numpy()
            else:
                pred, _ = postprocess(logits, conf_threshold=conf_thresh, min_pixels=min_px)
            n = min(pred.size, label_arr.size)
            preds_list.append(pred.flatten()[:n])
            labels_list.append(label_arr.flatten()[:n])

        miou = evaluate(preds_list, labels_list, label)

        if miou > best_miou:
            best_miou = miou
            best_config = (conf_thresh, min_px, label)

        # Save images for best config
        if args.save_images and best_config and best_config[2] == label:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(exist_ok=True)
            for (logits, name), label_arr in zip(all_logits, all_labels):
                pred, _ = postprocess(logits, conf_threshold=conf_thresh, min_pixels=min_px)
                try:
                    from PIL import Image
                    colors = [(0, 0, 0), (255, 255, 0), (0, 200, 255)]
                    def class_to_rgb(arr):
                        h, w = arr.shape
                        rgb = np.zeros((h, w, 3), dtype=np.uint8)
                        for c, color in enumerate(colors):
                            rgb[arr == c] = color
                        return rgb
                    pred_rgb = class_to_rgb(pred)
                    label_rgb = class_to_rgb(label_arr)
                    combined = np.concatenate([label_rgb, pred_rgb], axis=1)
                    Image.fromarray(combined).save(str(out_dir / f"{name}.png"))
                except Exception:
                    pass

    print(f"\n{'#' * 70}")
    print(f"  BEST CONFIG: {best_config[2]}")
    print(f"  mIoU: {best_miou:.4f}")
    print(f"  Threshold: {best_config[0]}, Min pixels: {best_config[1]}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
