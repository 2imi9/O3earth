#!/usr/bin/env python3
"""
remap_labels_3class.py — Remap 11-class labels to 3-class (bg, solar, wind).
Original: 0=bg, 1=solar, 2=wind, 3=gas, 4=coal, 5=nuclear, 6=hydro, 7=oil, 8=biomass, 9=geo, 10=storage
New:      0=bg, 1=solar, 2=wind (everything else -> 0)

Creates a backup of original labels and writes remapped labels in-place.
"""
import json
from pathlib import Path
import numpy as np
import tifffile

REMAP = {
    0: 0,    # background -> background
    1: 1,    # solar -> solar
    2: 2,    # wind -> wind
    3: 0,    # gas -> background
    4: 0,    # coal -> background
    5: 0,    # nuclear -> background
    6: 0,    # hydro -> background
    7: 0,    # oil -> background
    8: 0,    # biomass -> background
    9: 0,    # geothermal -> background
    10: 0,   # storage -> background
    255: 255, # nodata stays nodata
}

def main():
    dataset = Path("./dataset/windows/ready")
    remapped = 0
    backed_up = 0

    for win_dir in sorted(dataset.iterdir()):
        if not win_dir.is_dir():
            continue

        label_path = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_path.exists():
            continue

        arr = tifffile.imread(str(label_path))
        original_unique = set(np.unique(arr).tolist())

        # Skip if already only has 0, 1, 2, 255
        if original_unique <= {0, 1, 2, 255}:
            remapped += 1
            continue

        # Backup original
        backup_path = label_path.with_suffix(".tif.bak11class")
        if not backup_path.exists():
            tifffile.imwrite(str(backup_path), arr)
            backed_up += 1

        # Remap
        new_arr = np.zeros_like(arr)
        for old_val, new_val in REMAP.items():
            new_arr[arr == old_val] = new_val
        # Any value not in REMAP -> background
        for v in original_unique:
            if v not in REMAP:
                new_arr[arr == v] = 0

        tifffile.imwrite(str(label_path), new_arr)
        remapped += 1

        new_unique = set(np.unique(new_arr).tolist())
        has_solar = 1 in new_unique
        has_wind = 2 in new_unique
        if has_solar or has_wind:
            print(f"  {win_dir.name}: {original_unique} -> {new_unique}")

    print(f"\nRemapped {remapped} label files ({backed_up} backed up)")

    # Count windows per class
    solar_wins = 0
    wind_wins = 0
    for win_dir in sorted(dataset.iterdir()):
        if not win_dir.is_dir():
            continue
        label_path = win_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_path.exists():
            continue
        arr = tifffile.imread(str(label_path))
        unique = set(np.unique(arr).tolist())
        if 1 in unique:
            solar_wins += 1
        if 2 in unique:
            wind_wins += 1

    print(f"\nWindows with solar labels: {solar_wins}")
    print(f"Windows with wind labels: {wind_wins}")

if __name__ == "__main__":
    main()
