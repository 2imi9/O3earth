#!/usr/bin/env python3
"""
Extract OlmoEarth embeddings for each patch in the suitability dataset.

For each (lat, lon) in the dataset:
  1. Download a 128x128 Sentinel-2 patch (12 bands, 10m) from Planetary Computer
  2. Normalize per OlmoEarth spec
  3. Run through frozen OlmoEarth encoder
  4. Save mean-pooled 768-dim embedding

Output:
  - embeddings.npy  — shape (N, 768)
  - embeddings_meta.csv — metadata aligned with embeddings rows

Usage:
    python scripts/extract_embeddings.py \
        --dataset data/suitability_dataset.parquet \
        --output-dir data/embeddings \
        --model-size BASE \
        --batch-size 8 \
        --patch-size-px 128 \
        --workers 4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel-2 band info for OlmoEarth
# ---------------------------------------------------------------------------

# OlmoEarth expects these 12 bands in order:
# B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
OLMOEARTH_BANDS = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B8A",
    "B11", "B12", "B01", "B09",
]

# Planetary Computer Sentinel-2 L2A band names mapping
PC_BAND_MAP = {
    "B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08",
    "B05": "B05", "B06": "B06", "B07": "B07", "B8A": "B8A",
    "B11": "B11", "B12": "B12", "B01": "B01", "B09": "B09",
}

# Per-band normalization stats from OlmoEarth pretrain
# Format: (mean, std) — normalize as: (val - (mean - 2*std)) / (4*std)
# These are approximate; adjust if official stats differ.
BAND_STATS = {
    "B01": (1651.2, 783.0),
    "B02": (1349.3, 631.9),
    "B03": (1225.7, 568.5),
    "B04": (1187.3, 687.8),
    "B05": (1508.4, 609.7),
    "B06": (2365.9, 793.8),
    "B07": (2734.3, 932.3),
    "B08": (2597.5, 929.5),
    "B8A": (2897.0, 984.6),
    "B09": (846.5, 566.8),
    "B11": (2164.0, 862.0),
    "B12": (1497.7, 750.6),
}


# ---------------------------------------------------------------------------
# Planetary Computer tile fetching
# ---------------------------------------------------------------------------


def _get_sentinel2_patch(
    lat: float,
    lon: float,
    patch_size_px: int = 128,
    resolution: float = 10.0,
    max_cloud_pct: float = 20.0,
    time_range: str = "2023-01-01/2023-12-31",
) -> np.ndarray | None:
    """Fetch a Sentinel-2 L2A patch from Microsoft Planetary Computer.

    Returns array of shape (12, patch_size_px, patch_size_px) or None on failure.
    """
    try:
        import planetary_computer
        import pystac_client
        import stackstac
        from pyproj import Transformer
    except ImportError as e:
        logger.error(
            f"Missing dependency for Planetary Computer access: {e}. "
            "Install: uv pip install planetary-computer pystac-client stackstac pyproj"
        )
        return None

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        # Search with lat/lon bbox (slightly larger than needed)
        half_deg = 0.02
        search_bbox = [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=search_bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": max_cloud_pct}},
            max_items=5,
        )

        items = list(search.items())
        if not items:
            logger.debug(f"No Sentinel-2 items for ({lat:.4f}, {lon:.4f})")
            return None

        # Sort by cloud cover, use least cloudy
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        item = items[0]

        # Get native EPSG and compute UTM bounds
        epsg = item.properties.get("proj:epsg", 32611)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        cx, cy = transformer.transform(lon, lat)
        half_m = (patch_size_px * resolution) / 2.0
        bounds = [cx - half_m, cy - half_m, cx + half_m, cy + half_m]

        band_names_pc = [
            "B02", "B03", "B04", "B08",
            "B05", "B06", "B07", "B8A",
            "B11", "B12", "B01", "B09",
        ]

        stack = stackstac.stack(
            [item],
            assets=band_names_pc,
            resolution=resolution,
            bounds=bounds,
            epsg=epsg,
        )

        data = stack.compute()
        arr = data.values[0]  # (12, H, W)
        arr = np.nan_to_num(arr, nan=0.0)

        # Crop or pad to exact patch size
        _, h, w = arr.shape
        if h < patch_size_px or w < patch_size_px:
            padded = np.zeros((12, patch_size_px, patch_size_px), dtype=arr.dtype)
            padded[:, :min(h, patch_size_px), :min(w, patch_size_px)] = arr[
                :, :patch_size_px, :patch_size_px
            ]
            arr = padded
        else:
            y0 = (h - patch_size_px) // 2
            x0 = (w - patch_size_px) // 2
            arr = arr[:, y0 : y0 + patch_size_px, x0 : x0 + patch_size_px]

        return arr.astype(np.float32)

    except Exception as e:
        logger.debug(f"Failed to fetch patch for ({lat:.4f}, {lon:.4f}): {e}")
        return None


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Apply OlmoEarth normalization: (val - (mean - 2*std)) / (4*std).

    patch: shape (12, H, W), float32
    Returns normalized patch of same shape.
    """
    normalized = np.empty_like(patch)
    for i, band_name in enumerate(OLMOEARTH_BANDS):
        mean, std = BAND_STATS[band_name]
        normalized[i] = (patch[i] - (mean - 2 * std)) / (4 * std)
    return normalized


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_size: str, device: str):
    """Load OlmoEarth model as a frozen feature extractor."""
    model_id_map = {
        "NANO": "OLMOEARTH_V1_NANO",
        "TINY": "OLMOEARTH_V1_TINY",
        "BASE": "OLMOEARTH_V1_BASE",
        "LARGE": "OLMOEARTH_V1_LARGE",
    }
    model_id = model_id_map.get(model_size.upper())
    if model_id is None:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_id_map.keys())}")

    try:
        from rslearn.models.olmoearth_pretrain.model import OlmoEarth
        from olmoearth_pretrain_minimal.model_loader import ModelID as OlmoModelID

        # Must use the enum, not string, for HuggingFace download
        enum_id = OlmoModelID[model_id]
        model = OlmoEarth(
            model_id=enum_id,
            patch_size=4,
            selector=["encoder"],
        )
        model = model.to(device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        logger.info(f"Loaded OlmoEarth {model_id} on {device}")
        return model
    except ImportError:
        logger.error(
            "Cannot import OlmoEarth model. Make sure rslearn and olmoearth_pretrain are installed. "
            "pip install rslearn olmoearth-pretrain"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_embedding(model, patch: np.ndarray, device: str) -> np.ndarray:
    """Run a single normalized patch through the model, return 768-dim embedding.

    patch: (12, H, W) float32, already normalized
    Returns: (768,) float32 numpy array
    """
    from rslearn.train.model_context import ModelContext, RasterImage

    # OlmoEarth expects RasterImage with shape [C, T, H, W]
    tensor = torch.from_numpy(patch).unsqueeze(1).to(device)  # (12, 1, H, W)
    raster = RasterImage(image=tensor, timestamps=None, expected_timestamps=None)
    context = ModelContext(
        inputs=[{"sentinel2_l2a": raster}],
        metadatas=[],
        context_dict={},
    )

    # OlmoEarth returns FeatureMaps with .feature_maps list
    output = model(context)

    # Get feature tensor from FeatureMaps
    if hasattr(output, "feature_maps"):
        feat = output.feature_maps[0]  # [1, 768, H/patch, W/patch]
    elif isinstance(output, (list, tuple)):
        feat = output[-1]
    elif isinstance(output, dict):
        feat = list(output.values())[-1]
    else:
        feat = output

    # Global average pool: [1, C, H', W'] -> [1, C] -> [C]
    if feat.dim() == 4:
        embedding = feat.mean(dim=[2, 3]).squeeze(0)
    elif feat.dim() == 3:
        embedding = feat.mean(dim=1).squeeze(0)
    elif feat.dim() == 2:
        embedding = feat.squeeze(0)
    else:
        embedding = feat.flatten()

    return embedding.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract OlmoEarth embeddings for suitability dataset patches."
    )
    parser.add_argument(
        "--dataset",
        default="data/suitability_dataset.parquet",
        help="Path to suitability dataset (parquet or CSV).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/embeddings",
        help="Directory to save embeddings and metadata.",
    )
    parser.add_argument(
        "--model-size",
        default="BASE",
        choices=["NANO", "TINY", "BASE", "LARGE"],
        help="OlmoEarth model size.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (for future batched inference).")
    parser.add_argument("--patch-size-px", type=int, default=128, help="Patch size in pixels (128 = 1.28km at 10m).")
    parser.add_argument("--resolution", type=float, default=10.0, help="Sentinel-2 resolution in meters.")
    parser.add_argument("--time-range", default="2023-01-01/2023-12-31", help="Sentinel-2 time range.")
    parser.add_argument("--max-cloud-pct", type=float, default=20.0, help="Max cloud cover percentage.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers (future use).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial output.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per patch download.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Delay between retries (seconds).")

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "embeddings_meta.csv"
    progress_path = output_dir / "_progress.csv"  # tracks processed indices

    # --- Load dataset ---
    logger.info(f"Loading dataset from {dataset_path} ...")
    if str(dataset_path).endswith(".parquet"):
        df = pd.read_parquet(str(dataset_path))
    else:
        df = pd.read_csv(str(dataset_path))
    logger.info(f"  {len(df)} samples loaded.")

    # --- Determine device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # --- Load model ---
    model = _load_model(args.model_size, device)

    # --- Determine embedding dimension ---
    # Run a dummy input through the proper ModelContext interface
    from rslearn.train.model_context import ModelContext, RasterImage
    dummy_patch = np.random.randn(12, args.patch_size_px, args.patch_size_px).astype(np.float32)
    dummy_emb = _extract_embedding(model, dummy_patch, device)
    embed_dim = dummy_emb.shape[0]
    logger.info(f"Embedding dimension: {embed_dim}")
    del dummy_patch, dummy_emb

    # --- Resume support ---
    processed_indices = set()
    existing_embeddings = []
    existing_meta = []

    if args.resume and progress_path.exists():
        progress_df = pd.read_csv(str(progress_path))
        processed_indices = set(progress_df["index"].tolist())
        if embeddings_path.exists():
            existing_embeddings = list(np.load(str(embeddings_path)))
        if meta_path.exists():
            existing_meta_df = pd.read_csv(str(meta_path))
            existing_meta = existing_meta_df.to_dict("records")
        logger.info(f"Resuming: {len(processed_indices)} samples already processed.")

    # --- Extract embeddings ---
    all_embeddings = list(existing_embeddings)
    all_meta = list(existing_meta)
    failed_indices = []

    pbar = tqdm(total=len(df), desc="Extracting embeddings", unit=" patches",
                initial=len(processed_indices))

    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue

        lat, lon = row["lat"], row["lon"]

        # Download Sentinel-2 patch with retries
        patch = None
        for attempt in range(args.max_retries):
            patch = _get_sentinel2_patch(
                lat, lon,
                patch_size_px=args.patch_size_px,
                resolution=args.resolution,
                max_cloud_pct=args.max_cloud_pct,
                time_range=args.time_range,
            )
            if patch is not None:
                break
            if attempt < args.max_retries - 1:
                time.sleep(args.retry_delay)

        if patch is None:
            logger.debug(f"Skipping idx={idx} ({lat:.4f}, {lon:.4f}) — no data.")
            failed_indices.append(idx)
            pbar.update(1)
            continue

        # Normalize
        patch = _normalize_patch(patch)

        # Extract embedding
        try:
            embedding = _extract_embedding(model, patch, device)
        except Exception as e:
            logger.warning(f"Model inference failed for idx={idx}: {e}")
            failed_indices.append(idx)
            pbar.update(1)
            continue

        all_embeddings.append(embedding)
        all_meta.append({
            "index": int(idx),
            "lat": lat,
            "lon": lon,
            "energy_type": row.get("energy_type", ""),
            "label": int(row.get("label", -1)),
            "region": row.get("region", ""),
            "country_code": row.get("country_code", ""),
        })
        processed_indices.add(idx)

        # Periodic save: every 50 for the first 200, then every 500
        save_interval = 50 if len(all_embeddings) < 200 else 500
        if len(all_embeddings) % save_interval == 0:
            _save_checkpoint(all_embeddings, all_meta, processed_indices,
                             embeddings_path, meta_path, progress_path, embed_dim)

        pbar.update(1)

    pbar.close()

    # --- Final save ---
    _save_checkpoint(all_embeddings, all_meta, processed_indices,
                     embeddings_path, meta_path, progress_path, embed_dim)

    # --- Summary ---
    logger.info(f"\n=== Extraction Summary ===")
    logger.info(f"  Total samples:       {len(df)}")
    logger.info(f"  Successfully extracted: {len(all_embeddings)}")
    logger.info(f"  Failed / skipped:     {len(failed_indices)}")
    logger.info(f"  Embedding shape:      ({len(all_embeddings)}, {embed_dim})")
    logger.info(f"  Output directory:     {output_dir}")
    logger.info(f"  embeddings.npy:       {embeddings_path}")
    logger.info(f"  embeddings_meta.csv:  {meta_path}")

    if failed_indices:
        failed_path = output_dir / "failed_indices.txt"
        with open(str(failed_path), "w") as f:
            for fi in failed_indices:
                f.write(f"{fi}\n")
        logger.info(f"  Failed indices saved to: {failed_path}")

    logger.info("Done.")


def _save_checkpoint(embeddings, meta, processed_indices,
                     embeddings_path, meta_path, progress_path, embed_dim):
    """Save current state to disk."""
    if not embeddings:
        return
    arr = np.array(embeddings, dtype=np.float32)
    np.save(str(embeddings_path), arr)
    pd.DataFrame(meta).to_csv(str(meta_path), index=False)
    pd.DataFrame({"index": sorted(processed_indices)}).to_csv(str(progress_path), index=False)
    logger.debug(f"Checkpoint saved: {len(embeddings)} embeddings.")


if __name__ == "__main__":
    main()
