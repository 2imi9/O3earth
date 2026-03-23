#!/usr/bin/env python3
"""Restore all train_excluded windows back to train (in options field)."""
import json
from pathlib import Path

dataset = Path("./dataset/windows/ready")
restored = 0
for win_dir in sorted(dataset.iterdir()):
    if not win_dir.is_dir():
        continue
    meta_path = win_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("options", {}).get("split") == "train_excluded":
        meta["options"]["split"] = "train"
        meta_path.write_text(json.dumps(meta, indent=2))
        restored += 1
print(f"Restored {restored} windows back to options.split=train")
