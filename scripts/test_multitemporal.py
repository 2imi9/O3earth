#!/usr/bin/env python3
"""
Test multi-temporal (T=4) vs single-temporal (T=1) OlmoEarth embeddings.

Takes 200 random samples from the suitability dataset, fetches 4 seasonal
Sentinel-2 scenes per location, extracts embeddings with T=1 and T=4,
then compares classification performance.
"""

import numpy as np
import pandas as pd
import torch
import time
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Seasons: Q1 (winter), Q2 (spring), Q3 (summer), Q4 (fall)
SEASONS = [
    ("2023-01-01/2023-03-31", "Q1_winter"),
    ("2023-04-01/2023-06-30", "Q2_spring"),
    ("2023-07-01/2023-09-30", "Q3_summer"),
    ("2023-10-01/2023-12-31", "Q4_fall"),
]

# Band stats for normalization
OLMOEARTH_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
BAND_STATS = {
    "B01":(1651.2,783.0),"B02":(1349.3,631.9),"B03":(1225.7,568.5),"B04":(1187.3,687.8),
    "B05":(1508.4,609.7),"B06":(2365.9,793.8),"B07":(2734.3,932.3),"B08":(2597.5,929.5),
    "B8A":(2897.0,984.6),"B09":(846.5,566.8),"B11":(2164.0,862.0),"B12":(1497.7,750.6),
}


def fetch_patch(lat, lon, time_range, patch_size_px=128, resolution=10.0, max_cloud_pct=30.0):
    """Fetch one Sentinel-2 patch. Returns (12, H, W) or None."""
    try:
        import planetary_computer, pystac_client, stackstac
        from pyproj import Transformer
    except ImportError:
        return None

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        half_deg = 0.02
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[lon-half_deg, lat-half_deg, lon+half_deg, lat+half_deg],
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": max_cloud_pct}},
            max_items=3,
        )
        items = list(search.items())
        if not items:
            return None

        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        item = items[0]

        epsg = item.properties.get("proj:epsg", 32611)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        cx, cy = transformer.transform(lon, lat)
        half_m = (patch_size_px * resolution) / 2.0
        bounds = [cx-half_m, cy-half_m, cx+half_m, cy+half_m]

        bands = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
        stack = stackstac.stack([item], assets=bands, resolution=resolution, bounds=bounds, epsg=epsg)
        data = stack.compute()
        arr = np.nan_to_num(data.values[0], nan=0.0)

        _, h, w = arr.shape
        if h < patch_size_px or w < patch_size_px:
            padded = np.zeros((12, patch_size_px, patch_size_px), dtype=arr.dtype)
            padded[:, :min(h,patch_size_px), :min(w,patch_size_px)] = arr[:, :patch_size_px, :patch_size_px]
            arr = padded
        else:
            y0, x0 = (h-patch_size_px)//2, (w-patch_size_px)//2
            arr = arr[:, y0:y0+patch_size_px, x0:x0+patch_size_px]

        return arr.astype(np.float32)
    except Exception as e:
        logger.debug(f"Fetch failed ({lat:.2f},{lon:.2f}) {time_range}: {e}")
        return None


def normalize(patch):
    """OlmoEarth normalization."""
    out = np.empty_like(patch)
    for i, band in enumerate(OLMOEARTH_BANDS):
        mean, std = BAND_STATS[band]
        out[i] = (patch[i] - (mean - 2*std)) / (4*std)
    return out


@torch.no_grad()
def extract_embedding(model, tensor_CTHW, device):
    """Extract mean-pooled embedding from [C, T, H, W] tensor."""
    from rslearn.train.model_context import ModelContext, RasterImage

    raster = RasterImage(image=tensor_CTHW.to(device), timestamps=None, expected_timestamps=None)
    context = ModelContext(inputs=[{"sentinel2_l2a": raster}], metadatas=[], context_dict={})
    output = model(context)

    feat = output.feature_maps[0] if hasattr(output, "feature_maps") else output
    if feat.dim() == 4:
        return feat.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
    return feat.flatten().cpu().numpy()


def main():
    import os
    os.environ["RSLEARN_MULTIPROCESSING_CONTEXT"] = "spawn"

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "suitability_dataset_v2_shuffled.parquet"
    output_dir = project_root / "data" / "multitemporal_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(str(dataset_path))

    # Sample 200 diverse locations
    sample = df.groupby("energy_type").apply(
        lambda x: x.sample(min(50, len(x)), random_state=42)
    ).reset_index(drop=True)
    logger.info(f"Testing {len(sample)} samples")
    logger.info(f"Distribution:\n{sample['energy_type'].value_counts()}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth
    from olmoearth_pretrain_minimal.model_loader import ModelID
    model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_BASE, patch_size=4, selector=["encoder"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Model on {device}")

    emb_t1_list = []
    emb_t4_list = []
    meta_list = []
    skipped = 0

    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Multi-temporal test"):
        lat, lon = row["lat"], row["lon"]

        # Fetch 4 seasonal patches
        seasonal_patches = []
        for time_range, season_name in SEASONS:
            patch = None
            for attempt in range(2):
                patch = fetch_patch(lat, lon, time_range)
                if patch is not None:
                    break
                time.sleep(2)
            if patch is not None:
                seasonal_patches.append(normalize(patch))

        if len(seasonal_patches) < 2:
            skipped += 1
            continue

        # T=1: use first available season
        t1_patch = seasonal_patches[0]  # [12, 128, 128]
        t1_tensor = torch.from_numpy(t1_patch).unsqueeze(1)  # [12, 1, 128, 128]
        emb_t1 = extract_embedding(model, t1_tensor, device)

        # T=N: stack all available seasons
        t4_stack = np.stack(seasonal_patches, axis=1)  # [12, N, 128, 128]
        t4_tensor = torch.from_numpy(t4_stack)
        emb_t4 = extract_embedding(model, t4_tensor, device)

        emb_t1_list.append(emb_t1)
        emb_t4_list.append(emb_t4)
        meta_list.append({
            "lat": lat, "lon": lon,
            "energy_type": row["energy_type"],
            "label": int(row["label"]),
            "country_code": row.get("country_code", ""),
            "n_seasons": len(seasonal_patches),
        })

        # Checkpoint every 50
        if len(emb_t1_list) % 50 == 0:
            logger.info(f"Checkpoint: {len(emb_t1_list)} done, {skipped} skipped")
            np.save(str(output_dir / "emb_t1.npy"), np.array(emb_t1_list))
            np.save(str(output_dir / "emb_t4.npy"), np.array(emb_t4_list))
            pd.DataFrame(meta_list).to_csv(str(output_dir / "meta.csv"), index=False)

    # Final save
    np.save(str(output_dir / "emb_t1.npy"), np.array(emb_t1_list))
    np.save(str(output_dir / "emb_t4.npy"), np.array(emb_t4_list))
    pd.DataFrame(meta_list).to_csv(str(output_dir / "meta.csv"), index=False)

    logger.info(f"\n=== Results ===")
    logger.info(f"Extracted: {len(emb_t1_list)} | Skipped: {skipped}")
    logger.info(f"T=1 shape: {np.array(emb_t1_list).shape}")
    logger.info(f"T=4 shape: {np.array(emb_t4_list).shape}")

    # Quick comparison
    if len(emb_t1_list) >= 50:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        labels = np.array([m["label"] for m in meta_list])
        e1 = np.array(emb_t1_list)
        e4 = np.array(emb_t4_list)

        for name, emb in [("T=1", e1), ("T=multi", e4)]:
            model_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            n_splits = min(5, min(int(labels.sum()), int((labels==0).sum())))
            if n_splits < 2:
                logger.info(f"{name}: not enough class diversity")
                continue
            s = cross_val_score(model_lr, emb, labels, cv=n_splits, scoring="roc_auc")
            logger.info(f"{name}: AUC = {s.mean():.4f} +/- {s.std():.4f}")


if __name__ == "__main__":
    main()
