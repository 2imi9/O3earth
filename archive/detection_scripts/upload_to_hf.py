#!/usr/bin/env python3
"""
upload_to_hf.py — Upload fine-tuned OlmoEarth checkpoint and configs to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py --repo 2imi9/openenergyengine-olmoearth-base --create-repo
"""

import argparse
import os
import time
from pathlib import Path

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
datasets:
  - custom (EIA Form 860 + Sentinel-2 L2A)
pipeline_tag: image-segmentation
---

# OpenEnergy-Engine: OlmoEarth Energy Infrastructure Segmentation

Fine-tuned [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) (ViT-BASE) for
segmenting energy infrastructure from 12-band Sentinel-2 imagery, using EIA Form 860
power plant data as ground truth labels.

## Model Details

- **Base model**: OlmoEarth V1 BASE (89M encoder params)
- **Decoder**: UNetDecoder (16x16 to 128x128)
- **Task**: Semantic segmentation (11 classes)
- **Input**: 12-band Sentinel-2 L2A (B02-B12 + B01 + B09), 10m resolution
- **Patch size**: 8
- **Training**: Encoder frozen for first 10 epochs, then joint fine-tuning

## Classes

| ID | Class | EIA Source Codes |
|----|-------|-----------------|
| 0 | background | — |
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

## Usage with rslearn

```bash
# Clone and set up
git clone https://github.com/2imi9/OpenEnergy-Engine
cd OpenEnergy-Engine

# Download this checkpoint
huggingface-cli download 2imi9/openenergyengine-olmoearth-base \\
    --local-dir ./project_data

# Run prediction on a region
export MANAGEMENT_DIR=./project_data
rslearn model predict --config configs/model.yaml
```

## Training Data

- **Labels**: EIA Form 860 (2024) — ~13,000+ US power plants, capacity >= 1 MW
- **Imagery**: Sentinel-2 L2A via Microsoft Planetary Computer (May-Aug 2023)
- **Windows**: 9,292 materialized (8,113 train / 1,158 val)
- **Coverage**: Continental United States

## Training Details

- **Epochs**: 30 (encoder frozen for first 10)
- **Final val_loss**: 0.231
- **GPU**: NVIDIA RTX 5090
- **Framework**: rslearn + PyTorch Lightning

## Citation

```bibtex
@misc{openenergyengine2026,
  title={OpenEnergy-Engine: AI-Powered Earth Observation for Energy Verification},
  author={2imi9},
  year={2026},
  url={https://github.com/2imi9/OpenEnergy-Engine}
}
```
"""


def upload_with_retry(func, max_retries=5, delay=5, **kwargs):
    """Retry upload on network errors."""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            err = str(e)
            if attempt < max_retries - 1 and ("ConnectError" in err or "10054" in err or "ConnectionError" in err):
                print(f"  Attempt {attempt+1} failed ({err[:80]}...), retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--repo", default="2imi9/openenergyengine-olmoearth-base")
    parser.add_argument("--checkpoint", default="project_data/openenergyengine/run_00/best.ckpt")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--create-repo", action="store_true")
    args = parser.parse_args()

    # Force requests backend instead of httpx (fixes TLS issues on some Windows machines)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: pip install huggingface_hub")
        raise SystemExit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Find checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        for candidate in [
            Path("project_data/openenergyengine/run_00/best.ckpt"),
            Path("project_data/openenergyengine/run_00/last.ckpt"),
            Path("project_data/best.ckpt"),
        ]:
            if candidate.exists():
                ckpt_path = candidate
                print(f"Found checkpoint at {ckpt_path}")
                break
    if not ckpt_path.exists():
        print(f"Error: No checkpoint found at {ckpt_path}")
        raise SystemExit(1)

    print(f"Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.0f} MB)")

    # Create repo
    if args.create_repo:
        print(f"Creating repo {args.repo}...")
        upload_with_retry(api.create_repo, repo_id=args.repo, repo_type="model", exist_ok=True)
        print(f"  Repo ready")

    # Upload checkpoint
    print(f"Uploading checkpoint (this may take a few minutes)...")
    upload_with_retry(
        api.upload_file,
        path_or_fileobj=str(ckpt_path),
        path_in_repo=f"checkpoints/{ckpt_path.name}",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"  Checkpoint uploaded")

    # Upload configs
    configs_dir = Path(args.configs_dir)
    for fname in ["model.yaml", "dataset_config.json"]:
        fpath = configs_dir / fname
        if fpath.exists():
            print(f"Uploading {fname}...")
            upload_with_retry(
                api.upload_file,
                path_or_fileobj=str(fpath),
                path_in_repo=f"configs/{fname}",
                repo_id=args.repo,
                repo_type="model",
            )

    # Upload model card
    print("Uploading model card...")
    upload_with_retry(
        api.upload_file,
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"\nDone! Model uploaded to: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
