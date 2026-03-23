#!/usr/bin/env python3
"""
balance_dataset_3class.py - Balance for 3-class (bg, solar, wind).
Keep ALL solar/wind windows, subsample bg-only windows.
"""
import json
import random
from pathlib import Path
import numpy as np
import tifffile

def main():
    dataset = Path("./dataset/windows/ready")
    seed = 42
    random.seed(seed)

    solar_windows = []
    wind_windows = []
    bg_only_windows = []

    for win_dir in sorted(dataset.iterdir()):
        if not win_dir.is_dir():
            continue
        meta = json.loads((win_dir / "metadata.json").read_text())
        if meta.get("options", {}).get("split") != "train":
            continue

        label_path = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_path.exists():
            continue

        arr = tifffile.imread(str(label_path))
        unique = set(np.unique(arr).tolist())

        has_solar = 1 in unique
        has_wind = 2 in unique

        if has_solar:
            solar_windows.append(win_dir)
        elif has_wind:
            wind_windows.append(win_dir)
        else:
            bg_only_windows.append(win_dir)

    print(f"Solar windows: {len(solar_windows)}")
    print(f"Wind windows: {len(wind_windows)}")
    print(f"BG-only windows: {len(bg_only_windows)}")

    # Keep all solar + wind, subsample bg to match ~1:1 with minority total
    minority_total = len(solar_windows) + len(wind_windows)
    # Keep same number of bg as minority windows (1:1 ratio)
    n_bg_keep = min(minority_total, len(bg_only_windows))

    random.shuffle(bg_only_windows)
    bg_keep = bg_only_windows[:n_bg_keep]
    bg_exclude = bg_only_windows[n_bg_keep:]

    print(f"\nBG kept: {n_bg_keep}")
    print(f"BG excluded: {len(bg_exclude)}")
    print(f"New training set: {minority_total + n_bg_keep}")
    print(f"  Solar: {len(solar_windows)} ({len(solar_windows)/(minority_total+n_bg_keep)*100:.1f}%)")
    print(f"  Wind:  {len(wind_windows)} ({len(wind_windows)/(minority_total+n_bg_keep)*100:.1f}%)")
    print(f"  BG:    {n_bg_keep} ({n_bg_keep/(minority_total+n_bg_keep)*100:.1f}%)")

    changed = 0
    for win_dir in bg_exclude:
        meta_path = win_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["options"]["split"] = "train_excluded"
        meta_path.write_text(json.dumps(meta, indent=2))
        changed += 1

    print(f"\nExcluded {changed} bg-only windows")

if __name__ == "__main__":
    main()
