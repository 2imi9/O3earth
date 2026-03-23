#!/usr/bin/env python3
"""
Resplit windows into train/val with balanced sampling.

Strategy:
  - All 592 windows with labels: 85% train, 15% val
  - Add a small number of background-only windows (2x labeled count)
  - This gives ~60% labeled windows in each split

This prevents the model from being overwhelmed by background-only examples.
"""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import tifffile


def main():
    root = Path("dataset/windows/ready")

    labeled = []
    unlabeled = []

    print("Scanning windows for label content...")
    for i, win in enumerate(sorted(root.iterdir())):
        if not win.is_dir():
            continue
        label_tif = win / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_tif.exists():
            continue

        lbl = tifffile.imread(str(label_tif))
        if lbl.ndim == 3:
            lbl = lbl[0]

        if lbl.max() > 0:
            labeled.append(win)
        else:
            unlabeled.append(win)

        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i + 1} windows...")

    print(f"\nLabeled windows: {len(labeled)}")
    print(f"Unlabeled windows: {len(unlabeled)}")

    # Deterministic shuffle
    rng = np.random.RandomState(42)
    labeled_idx = rng.permutation(len(labeled))
    unlabeled_idx = rng.permutation(len(unlabeled))

    # Split labeled: 85/15
    n_labeled_train = int(0.85 * len(labeled))
    labeled_train = [labeled[i] for i in labeled_idx[:n_labeled_train]]
    labeled_val = [labeled[i] for i in labeled_idx[n_labeled_train:]]

    # Add 2x background windows (so ~33% of training is background)
    n_bg_train = min(2 * n_labeled_train, len(unlabeled))
    n_bg_val = min(2 * len(labeled_val), len(unlabeled) - n_bg_train)
    bg_train = [unlabeled[i] for i in unlabeled_idx[:n_bg_train]]
    bg_val = [unlabeled[i] for i in unlabeled_idx[n_bg_train:n_bg_train + n_bg_val]]

    train_windows = labeled_train + bg_train
    val_windows = labeled_val + bg_val

    print(f"\nTrain: {len(train_windows)} ({len(labeled_train)} labeled + {len(bg_train)} bg)")
    print(f"Val:   {len(val_windows)} ({len(labeled_val)} labeled + {len(bg_val)} bg)")

    # Write split tags
    all_ready = set(w.name for w in root.iterdir() if w.is_dir())
    train_names = set(w.name for w in train_windows)
    val_names = set(w.name for w in val_windows)

    # Clear all splits first, then set train/val
    updated = 0
    for win in sorted(root.iterdir()):
        if not win.is_dir():
            continue
        meta_path = win / "metadata.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        if "tags" not in meta:
            meta["tags"] = {}

        if win.name in train_names:
            meta["tags"]["split"] = "train"
            updated += 1
        elif win.name in val_names:
            meta["tags"]["split"] = "val"
            updated += 1
        else:
            # Remove from training entirely (not in balanced set)
            meta["tags"]["split"] = "unused"
            updated += 1

        meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nUpdated {updated} window metadata files")
    print(f"Windows marked 'unused': {updated - len(train_windows) - len(val_windows)}")
    print("Done!")


if __name__ == "__main__":
    main()
