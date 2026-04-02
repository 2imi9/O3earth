#!/usr/bin/env python3
"""
Extract OlmoEarth embeddings for each patch in the suitability dataset.

For each (lat, lon) in the dataset:
  1. Download a 128x128 Sentinel-2 patch (12 bands, 10m) from Planetary Computer
  2. Normalize per OlmoEarth spec
  3. Run through frozen OlmoEarth encoder
  4. Save mean-pooled 768-dim embedding
  5. For solar/wind/hydro: fetch 4 seasonal scenes (T=4) for richer temporal signal
     For geothermal: single scene (T=1) as temporal variation hurts static geology tasks

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


# Seasons for multi-temporal extraction
SEASONS = [
    "2023-01-01/2023-03-31",  # Q1 winter
    "2023-04-01/2023-06-30",  # Q2 spring
    "2023-07-01/2023-09-30",  # Q3 summer
    "2023-10-01/2023-12-31",  # Q4 fall
]

# Energy types that benefit from multi-temporal (T=4)
MULTI_TEMPORAL_TYPES = {"solar", "wind", "hydro"}


def _fetch_one_season(lat, lon, season_range, patch_size_px, resolution, max_cloud_pct, max_retries, retry_delay):
    """Helper for parallel season fetch."""
    for attempt in range(max_retries):
        patch = _get_sentinel2_patch(lat, lon, patch_size_px, resolution, max_cloud_pct, season_range)
        if patch is not None:
            return patch
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return None


def _get_multitemporal_patch(
    lat: float, lon: float,
    patch_size_px: int = 128, resolution: float = 10.0,
    max_cloud_pct: float = 30.0, max_retries: int = 2, retry_delay: float = 2.0,
) -> np.ndarray | None:
    """Fetch 4 seasonal Sentinel-2 patches in parallel, return (12, T, H, W) or None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                _fetch_one_season, lat, lon, sr,
                patch_size_px, resolution, max_cloud_pct, max_retries, retry_delay
            ): sr for sr in SEASONS
        }
        patches = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                patches.append(result)

    if len(patches) < 2:
        return None

    return np.stack(patches, axis=1)


@torch.no_grad()
def _extract_embedding(model, patch: np.ndarray, device: str) -> np.ndarray:
    """Run patch through the model, return mean-pooled 768-dim embedding.

    patch: (12, H, W) for T=1 or (12, T, H, W) for T>1, float32, already normalized
    Returns: (768,) float32 numpy array
    """
    from rslearn.train.model_context import ModelContext, RasterImage

    # Handle both T=1 and T>1
    if patch.ndim == 3:
        # (12, H, W) -> (12, 1, H, W)
        tensor = torch.from_numpy(patch).unsqueeze(1).to(device)
    else:
        # Already (12, T, H, W)
        tensor = torch.from_numpy(patch).to(device)

    raster = RasterImage(image=tensor, timestamps=None, expected_timestamps=None)
    context = ModelContext(
        inputs=[{"sentinel2_l2a": raster}],
        metadatas=[],
        context_dict={},
    )

    output = model(context)

    if hasattr(output, "feature_maps"):
        feat = output.feature_maps[0]
    elif isinstance(output, (list, tuple)):
        feat = output[-1]
    elif isinstance(output, dict):
        feat = list(output.values())[-1]
    else:
        feat = output

    # Mean pooling (optimal for binary classification per sufficient statistics theory)
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

    from concurrent.futures import ThreadPoolExecutor

    def _fetch_patch_for_row(row_idx, row):
        """Fetch patch (T=1 or T=4) for a single row."""
        lat, lon = row["lat"], row["lon"]
        energy_type = row.get("energy_type", "solar")
        if energy_type in MULTI_TEMPORAL_TYPES:
            return _get_multitemporal_patch(
                lat, lon,
                patch_size_px=args.patch_size_px,
                resolution=args.resolution,
                max_cloud_pct=args.max_cloud_pct,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
        else:
            for attempt in range(args.max_retries):
                patch = _get_sentinel2_patch(
                    lat, lon,
                    patch_size_px=args.patch_size_px,
                    resolution=args.resolution,
                    max_cloud_pct=args.max_cloud_pct,
                    time_range=args.time_range,
                )
                if patch is not None:
                    return patch
                if attempt < args.max_retries - 1:
                    time.sleep(args.retry_delay)
            return None

    # Build list of rows to process
    pending_rows = [(idx, row) for idx, row in df.iterrows() if idx not in processed_indices]

    pbar = tqdm(total=len(df), desc="Extracting embeddings", unit=" patches",
                initial=len(processed_indices))

    # Prefetch pipeline: download next batch while GPU processes current
    prefetch_workers = 4  # concurrent samples being downloaded
    with ThreadPoolExecutor(max_workers=prefetch_workers) as prefetch_pool:
        # Submit first batch
        future_to_row = {}
        row_iter = iter(pending_rows)

        # Fill the prefetch queue
        for _ in range(prefetch_workers):
            try:
                idx, row = next(row_iter)
                future = prefetch_pool.submit(_fetch_patch_for_row, idx, row)
                future_to_row[future] = (idx, row)
            except StopIteration:
                break

        from concurrent.futures import as_completed
        while future_to_row:
            # Wait for any download to complete
            done_futures = []
            for f in list(future_to_row.keys()):
                if f.done():
                    done_futures.append(f)

            if not done_futures:
                # Wait for at least one
                done = next(as_completed(future_to_row))
                done_futures = [done]

            for future in done_futures:
                idx, row = future_to_row.pop(future)
                patch = future.result()

                # Submit next download immediately
                try:
                    next_idx, next_row = next(row_iter)
                    next_future = prefetch_pool.submit(_fetch_patch_for_row, next_idx, next_row)
                    future_to_row[next_future] = (next_idx, next_row)
                except StopIteration:
                    pass

                if patch is None:
                    logger.debug(f"Skipping idx={idx} — no data.")
                    failed_indices.append(idx)
                    pbar.update(1)
                    continue

                # Normalize
                if patch.ndim == 3:
                    patch = _normalize_patch(patch)
                else:
                    for t in range(patch.shape[1]):
                        patch[:, t, :, :] = _normalize_patch(patch[:, t, :, :])

                # Extract embedding (GPU)
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
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "energy_type": row.get("energy_type", ""),
                    "label": int(row.get("label", -1)),
                    "region": row.get("region", ""),
                    "country_code": row.get("country_code", ""),
                })
                processed_indices.add(idx)

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
