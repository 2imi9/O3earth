#!/usr/bin/env python3
"""Move prepared windows (those with items.json) from default/ to ready/ group.

Filesystem move on same drive is instant — no data copied.
After running, use --group ready for ingest/materialize/train.
"""
import os
import shutil
import json
from pathlib import Path

ROOT = Path("dataset/windows")
SRC = ROOT / "default"
DST = ROOT / "ready"


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} does not exist")
        return

    DST.mkdir(parents=True, exist_ok=True)

    prepared = 0
    skipped = 0
    errors = 0

    entries = list(os.scandir(SRC))
    total = len(entries)
    print(f"Scanning {total} windows in {SRC}...")

    for i, entry in enumerate(entries):
        if not entry.is_dir():
            continue

        items_path = os.path.join(entry.path, "items.json")
        if not os.path.exists(items_path):
            skipped += 1
            continue

        # Move to ready group
        dst_path = DST / entry.name
        try:
            shutil.move(entry.path, dst_path)

            # Update the group in metadata.json
            meta_path = dst_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["group"] = "ready"
                with open(meta_path, "w") as f:
                    json.dump(meta, f)

            prepared += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error moving {entry.name}: {e}")

        if (i + 1) % 10000 == 0:
            print(f"  [{i+1}/{total}] {prepared} moved, {skipped} skipped, {errors} errors")

    print(f"\nDone!")
    print(f"  Moved to ready/: {prepared}")
    print(f"  Skipped (not prepared): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Remaining in default/: {skipped}")


if __name__ == "__main__":
    main()
