#!/usr/bin/env python3
"""Upload to HuggingFace using requests library (bypasses httpx TLS issues)."""

import os
import json
import hashlib
import requests
from pathlib import Path

TOKEN = os.environ.get("HF_TOKEN", "")
REPO = "2imi9/openenergyengine-olmoearth-base"
API = "https://huggingface.co/api"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

MODEL_CARD = """---
license: mit
tags:
  - olmoearth
  - segmentation
  - satellite
  - energy-infrastructure
  - sentinel-2
  - eia-860
  - geospatial
pipeline_tag: image-segmentation
---

# OpenEnergy-Engine: OlmoEarth Energy Infrastructure Segmentation

Fine-tuned [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) (ViT-BASE) for
segmenting energy infrastructure from 12-band Sentinel-2 imagery, using EIA Form 860
power plant data as ground truth labels.

## Model Details

- **Base model**: OlmoEarth V1 BASE (89M encoder params)
- **Decoder**: UNetDecoder
- **Task**: Semantic segmentation (11 classes)
- **Input**: 12-band Sentinel-2 L2A, 10m resolution
- **Training**: 30 epochs, encoder frozen first 10, RTX 5090
- **val_loss**: 0.231

## Classes

| ID | Class | EIA Source Codes |
|----|-------|-----------------|
| 0 | background | - |
| 1 | solar | SUN |
| 2 | wind | WND |
| 3 | gas | NG, BFG, SGC, SGP, OG |
| 4 | coal | COL, PC, WC |
| 5 | nuclear | NUC |
| 6 | hydro | WAT |
| 7 | oil | OIL, DFO, RFO, TDF |
| 8 | biomass | WDS, LFG, OBG, OBL, OBS, AB, MSW |
| 9 | geothermal | GEO |
| 10 | storage | MWH |

## Training Data

- **Labels**: EIA Form 860 (2024) - 13,000+ US power plants
- **Imagery**: Sentinel-2 L2A via Microsoft Planetary Computer (May-Aug 2023)
- **Windows**: 8,113 train / 1,158 val (128x128 @ 10m)
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_repo():
    print("Creating repo...")
    r = requests.post(
        f"{API}/repos/create",
        headers=HEADERS,
        json={"type": "model", "name": "openenergyengine-olmoearth-base", "private": False},
    )
    if r.status_code == 200:
        print("  Repo created!")
    elif r.status_code == 409:
        print("  Repo already exists")
    else:
        print(f"  Status {r.status_code}: {r.text[:200]}")


def upload_small_file(local_path, repo_path, content=None):
    """Upload a small file (<10MB) via the commit API."""
    if content is None:
        content = Path(local_path).read_text(encoding="utf-8")

    print(f"Uploading {repo_path}...")
    r = requests.post(
        f"{API}/models/{REPO}/commit/main",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={
            "summary": f"Upload {repo_path}",
            "files": [{"path": repo_path, "content": content}],
        },
    )
    if r.status_code in (200, 201):
        print(f"  Done!")
    else:
        print(f"  Status {r.status_code}: {r.text[:300]}")


def upload_large_file(local_path, repo_path):
    """Upload large file via LFS."""
    path = Path(local_path)
    size = path.stat().st_size
    file_sha = sha256(local_path)

    print(f"Uploading {repo_path} ({size/1e6:.0f} MB) via LFS...")

    # Step 1: Request upload URL
    r = requests.post(
        f"{API}/models/{REPO}/preupload/main",
        headers=HEADERS,
        json={"files": [{"path": repo_path, "sample": file_sha[:8], "size": size}]},
    )
    if r.status_code != 200:
        print(f"  Preupload failed: {r.status_code} {r.text[:300]}")
        return False

    preupload = r.json()
    print(f"  Preupload response: needs upload = {preupload}")

    # Step 2: LFS batch request
    lfs_url = f"https://huggingface.co/{REPO}.git/info/lfs/objects/batch"
    oid = f"sha256:{file_sha}"
    r = requests.post(
        lfs_url,
        headers={**HEADERS, "Content-Type": "application/vnd.git-lfs+json"},
        json={
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{"oid": file_sha, "size": size}],
        },
    )
    if r.status_code != 200:
        print(f"  LFS batch failed: {r.status_code} {r.text[:300]}")
        return False

    batch = r.json()
    obj = batch["objects"][0]

    if "actions" in obj:
        upload_action = obj["actions"]["upload"]
        upload_url = upload_action["href"]
        upload_headers = upload_action.get("header", {})

        print(f"  Uploading to LFS ({size/1e6:.0f} MB)...")
        with open(local_path, "rb") as f:
            r = requests.put(
                upload_url,
                headers=upload_headers,
                data=f,
            )
        if r.status_code in (200, 201):
            print(f"  LFS upload complete!")
        else:
            print(f"  LFS upload failed: {r.status_code}")
            return False
    else:
        print(f"  File already exists in LFS")

    # Step 3: Create commit referencing the LFS file
    lfs_pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{file_sha}\nsize {size}\n"
    r = requests.post(
        f"{API}/models/{REPO}/commit/main",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={
            "summary": f"Upload {repo_path}",
            "lfsFiles": [{"path": repo_path, "algo": "sha256", "oid": file_sha, "size": size}],
            "files": [],
        },
    )
    if r.status_code in (200, 201):
        print(f"  Commit created!")
        return True
    else:
        print(f"  Commit failed: {r.status_code} {r.text[:300]}")
        return False


def main():
    if not TOKEN:
        print("Set HF_TOKEN environment variable")
        return

    print(f"Token: {TOKEN[:10]}...")

    # 1. Create repo
    create_repo()

    # 2. Upload README
    upload_small_file(None, "README.md", content=MODEL_CARD)

    # 3. Upload configs
    for fname in ["model.yaml", "dataset_config.json"]:
        fpath = Path("configs") / fname
        if fpath.exists():
            upload_small_file(str(fpath), f"configs/{fname}")

    # 4. Upload checkpoint (large file, uses LFS)
    ckpt = Path("project_data/openenergyengine/run_00/best.ckpt")
    if ckpt.exists():
        upload_large_file(str(ckpt), "checkpoints/best.ckpt")
    else:
        print(f"Checkpoint not found: {ckpt}")

    print(f"\nDone! Check: https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
