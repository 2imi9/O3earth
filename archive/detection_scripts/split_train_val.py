#!/usr/bin/env python3
"""
split_train_val.py — Assign train/val split tags to rslearn windows.

Uses a deterministic hash-based split so results are reproducible.
Default: 87.5% train, 12.5% val.
"""

import argparse
import hashlib
import json
from pathlib import Path


def hash_split(name: str, val_fraction: float = 0.125) -> str:
    """Deterministic split based on window directory name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < (val_fraction * 1000) else "train"


def main():
    parser = argparse.ArgumentParser(description="Split rslearn windows into train/val")
    parser.add_argument("--root", default="./dataset", help="rslearn dataset root")
    parser.add_argument("--group", default="default", help="Window group name")
    parser.add_argument("--val-fraction", type=float, default=0.125, help="Fraction for validation")
    args = parser.parse_args()

    windows_dir = Path(args.root) / "windows" / args.group
    if not windows_dir.exists():
        print(f"Error: {windows_dir} does not exist. Run create_windows.sh and build_dataset.sh first.")
        raise SystemExit(1)

    train_count = 0
    val_count = 0

    for window_dir in sorted(windows_dir.iterdir()):
        if not window_dir.is_dir():
            continue

        split = hash_split(window_dir.name, args.val_fraction)

        # rslearn uses a metadata.json file in each window for tags
        meta_path = window_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}

        if "tags" not in meta:
            meta["tags"] = {}
        meta["tags"]["split"] = split

        meta_path.write_text(json.dumps(meta, indent=2))

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    total = train_count + val_count
    print(f"Split complete: {train_count} train, {val_count} val ({total} total)")
    print(f"  Train: {train_count/total*100:.1f}%")
    print(f"  Val:   {val_count/total*100:.1f}%")
    print(f"\nNext step: bash scripts/train.sh")


if __name__ == "__main__":
    main()
