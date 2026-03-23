#!/usr/bin/env python3
"""
balance_dataset.py — Subsample background-only windows to balance training set.
rslearn uses metadata.json "options" field (NOT "tags") for split filtering.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import tifffile


def main():
    dataset = Path("./dataset/windows/ready")
    keep_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    seed = 42
    random.seed(seed)

    bg_only_windows = []
    minority_windows = []

    for win_dir in sorted(dataset.iterdir()):
        if not win_dir.is_dir():
            continue
        meta_path = win_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        # rslearn uses "options" not "tags" for split filtering
        split = meta.get("options", {}).get("split", "")
        if split != "train":
            continue

        label_path = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_path.exists():
            continue

        arr = tifffile.imread(str(label_path))
        unique = set(np.unique(arr).tolist())
        non_bg = unique - {0, 255}

        if non_bg:
            minority_windows.append(win_dir)
        else:
            bg_only_windows.append(win_dir)

    n_keep = int(len(bg_only_windows) * keep_ratio)
    random.shuffle(bg_only_windows)
    keep = bg_only_windows[:n_keep]
    exclude = bg_only_windows[n_keep:]

    print(f"Minority-class windows (kept): {len(minority_windows)}")
    print(f"BG-only windows total: {len(bg_only_windows)}")
    print(f"BG-only kept ({keep_ratio*100:.0f}%): {n_keep}")
    print(f"BG-only excluded: {len(exclude)}")
    print(f"New training set size: {len(minority_windows) + n_keep}")

    # Update OPTIONS for excluded windows
    changed = 0
    for win_dir in exclude:
        meta_path = win_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["options"]["split"] = "train_excluded"
        meta_path.write_text(json.dumps(meta, indent=2))
        changed += 1

    print(f"Updated {changed} windows options.split to 'train_excluded'")


if __name__ == "__main__":
    main()
