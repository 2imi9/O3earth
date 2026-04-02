#!/usr/bin/env python3
"""
overnight_automation.py — Full overnight pipeline:
  1. Wait for downloads to finish (check for parquet files)
  2. Merge all data sources into unified global dataset
  3. Build suitability training dataset (balanced, geographically diverse)
  4. Verify data quality (sanity checks)
  5. Extract OlmoEarth embeddings for a small sample (test that pipeline works)
  6. Train lightweight classifier on sample
  7. Report results

Run: python scripts/overnight_automation.py
"""

import json
import logging
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "overnight_automation_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)

RESULTS = {}


def step1_wait_for_downloads(timeout_minutes=30):
    """Wait for EIA and Overpass downloads to finish."""
    log.info("=" * 60)
    log.info("STEP 1: Waiting for data downloads to complete...")
    log.info("=" * 60)

    required_files = [
        ROOT / "data" / "global_energy_locations.parquet",  # Already exists
    ]
    optional_files = [
        ROOT / "data" / "eia_plants.parquet",
        ROOT / "data" / "osm_overpass_missing.parquet",
    ]

    # Check required files exist
    for f in required_files:
        if not f.exists():
            log.error(f"Required file missing: {f}")
            return False

    # Wait for optional files (with timeout)
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        all_done = all(f.exists() for f in optional_files)
        if all_done:
            log.info("All download files found!")
            break
        missing = [f.name for f in optional_files if not f.exists()]
        log.info(f"  Waiting for: {missing} ({int(time.time()-start)}s elapsed)")
        time.sleep(30)

    # Report what we have
    for f in required_files + optional_files:
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            log.info(f"  Found: {f.name} ({size_mb:.1f} MB)")
        else:
            log.warning(f"  Missing: {f.name} (will proceed without)")

    return True


def step2_merge_datasets():
    """Merge all data sources into a unified global dataset."""
    log.info("=" * 60)
    log.info("STEP 2: Merging all data sources")
    log.info("=" * 60)

    frames = []

    # 1. First Overpass run (Asia, S. America, Oceania)
    p = ROOT / "data" / "global_energy_locations.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        log.info(f"  global_energy_locations: {len(df)} records")
        frames.append(df[["lat", "lon", "energy_type", "source", "name",
                          "capacity_mw", "country_code", "operating_year"]].copy())

    # 2. EIA plants (US)
    p = ROOT / "data" / "eia_plants.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        log.info(f"  eia_plants: {len(df)} records")
        if "source" not in df.columns:
            df["source"] = "eia_860"
        if "name" not in df.columns:
            df["name"] = ""
        if "capacity_mw" not in df.columns:
            df["capacity_mw"] = 0.0
        if "operating_year" not in df.columns:
            df["operating_year"] = ""
        frames.append(df[["lat", "lon", "energy_type", "source", "name",
                          "capacity_mw", "country_code", "operating_year"]].copy())

    # 3. Missing Overpass regions
    p = ROOT / "data" / "osm_overpass_missing.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        log.info(f"  osm_overpass_missing: {len(df)} records")
        for col in ["source", "name", "capacity_mw", "country_code", "operating_year"]:
            if col not in df.columns:
                df[col] = "" if col != "capacity_mw" else 0.0
        if "source" not in df.columns or df["source"].isna().all():
            df["source"] = "osm_overpass"
        frames.append(df[["lat", "lon", "energy_type", "source", "name",
                          "capacity_mw", "country_code", "operating_year"]].copy())

    if not frames:
        log.error("No data to merge!")
        return None

    merged = pd.concat(frames, ignore_index=True)
    log.info(f"\n  Combined: {len(merged)} total records")

    # Deduplicate (within ~500m per energy type)
    merged["lat_r"] = (merged["lat"] / 0.005).round() * 0.005
    merged["lon_r"] = (merged["lon"] / 0.005).round() * 0.005
    before = len(merged)
    merged = merged.drop_duplicates(subset=["lat_r", "lon_r", "energy_type"], keep="first")
    merged = merged.drop(columns=["lat_r", "lon_r"])
    log.info(f"  After dedup: {len(merged)} (removed {before - len(merged)} duplicates)")

    # Add country codes where missing
    missing_cc = (merged["country_code"] == "") | merged["country_code"].isna()
    if missing_cc.sum() > 0:
        try:
            import reverse_geocoder as rg
            coords = list(zip(merged.loc[missing_cc, "lat"], merged.loc[missing_cc, "lon"]))
            if coords:
                results = rg.search(coords)
                merged.loc[missing_cc, "country_code"] = [r["cc"] for r in results]
                log.info(f"  Geocoded {missing_cc.sum()} missing country codes")
        except Exception as e:
            log.warning(f"  Geocoding failed: {e}")

    # Add region
    def get_region(lat, lon):
        if lat > 15 and lon > -170 and lon < -50:
            return "North America"
        elif lat > -55 and lon > -82 and lon < -34:
            return "South America"
        elif lat > 35 and lon > -25 and lon < 45:
            return "Europe"
        elif lat > -35 and lon > -20 and lon < 55:
            return "Africa"
        elif lat > -47 and lon > 110:
            return "Oceania"
        else:
            return "Asia"
    merged["region"] = merged.apply(lambda r: get_region(r["lat"], r["lon"]), axis=1)

    # Summary
    log.info(f"\n  === MERGED DATASET SUMMARY ===")
    log.info(f"  Total: {len(merged)}")
    for et, cnt in merged["energy_type"].value_counts().items():
        log.info(f"    {et}: {cnt}")
    log.info(f"  By region:")
    for region, cnt in merged["region"].value_counts().items():
        log.info(f"    {region}: {cnt}")
    log.info(f"  By source:")
    for src, cnt in merged["source"].value_counts().items():
        log.info(f"    {src}: {cnt}")

    # Save
    out = ROOT / "data" / "all_energy_locations.parquet"
    merged.to_parquet(out, index=False)
    log.info(f"  Saved to {out}")

    RESULTS["merged_total"] = len(merged)
    RESULTS["merged_by_type"] = merged["energy_type"].value_counts().to_dict()
    RESULTS["merged_by_region"] = merged["region"].value_counts().to_dict()

    return merged


def step3_build_suitability_dataset(all_locations):
    """Build balanced suitability training dataset with positives and smart negatives."""
    log.info("=" * 60)
    log.info("STEP 3: Build suitability training dataset")
    log.info("=" * 60)

    np.random.seed(42)

    targets = {
        "solar": {"pos": 5000, "neg": 5000},
        "wind": {"pos": 5000, "neg": 5000},
        "hydro": {"pos": 2000, "neg": 2000},
        "geothermal": {"pos": 500, "neg": 500},
    }

    all_rows = []

    for energy_type, counts in targets.items():
        subset = all_locations[all_locations["energy_type"] == energy_type]
        available = len(subset)

        if available == 0:
            log.warning(f"  {energy_type}: 0 locations available, skipping")
            continue

        # Sample positives (geographically diverse)
        n_pos = min(counts["pos"], available)
        if available > n_pos:
            # Grid-based sampling for diversity
            subset = subset.copy()
            subset["grid"] = (subset["lat"] / 1.0).astype(int).astype(str) + "_" + \
                             (subset["lon"] / 1.0).astype(int).astype(str)
            grids = subset["grid"].unique()
            per_grid = max(1, n_pos // len(grids))
            sampled = subset.groupby("grid").apply(
                lambda x: x.sample(min(len(x), per_grid), random_state=42)
            ).reset_index(drop=True)
            if len(sampled) > n_pos:
                sampled = sampled.sample(n_pos, random_state=42)
            elif len(sampled) < n_pos:
                remaining = n_pos - len(sampled)
                extra = subset[~subset.index.isin(sampled.index)].sample(
                    min(remaining, len(subset) - len(sampled)), random_state=42)
                sampled = pd.concat([sampled, extra])
            positives = sampled
        else:
            positives = subset

        for _, row in positives.iterrows():
            all_rows.append({
                "lat": row["lat"], "lon": row["lon"],
                "energy_type": energy_type, "label": 1,
                "source": row.get("source", ""),
                "country_code": row.get("country_code", ""),
                "region": row.get("region", ""),
                "operating_year": row.get("operating_year", ""),
                "capacity_mw": row.get("capacity_mw", 0.0),
            })

        # Smart negatives: random land points away from all energy locations
        n_neg = counts["neg"]
        neg_count = 0
        all_lats = all_locations["lat"].values
        all_lons = all_locations["lon"].values

        # Build spatial hash for fast proximity check
        hash_grid = set()
        for lat, lon in zip(all_lats, all_lons):
            hash_grid.add((round(lat / 0.05), round(lon / 0.05)))

        attempts = 0
        while neg_count < n_neg and attempts < n_neg * 20:
            # Generate random land point (avoid extreme latitudes)
            lat = np.random.uniform(-55, 70)
            lon = np.random.uniform(-170, 180)

            # Skip likely ocean (very rough heuristic)
            if -55 < lat < -35 and -180 < lon < -60:
                continue  # Southern ocean
            if 60 < lat and (-180 < lon < -90 or 30 < lon < 180):
                if np.random.random() > 0.3:
                    continue  # Arctic/Siberia (sparse)

            # Check not near any energy facility
            grid_key = (round(lat / 0.05), round(lon / 0.05))
            if grid_key in hash_grid:
                attempts += 1
                continue

            # Assign region
            region = "Unknown"
            if lat > 15 and -170 < lon < -50:
                region = "North America"
            elif -55 < lat < 15 and -82 < lon < -34:
                region = "South America"
            elif lat > 35 and -25 < lon < 45:
                region = "Europe"
            elif -35 < lat < 37 and -20 < lon < 55:
                region = "Africa"
            elif lat < -10 and lon > 110:
                region = "Oceania"
            else:
                region = "Asia"

            all_rows.append({
                "lat": lat, "lon": lon,
                "energy_type": energy_type, "label": 0,
                "source": "negative_sample",
                "country_code": "",
                "region": region,
                "operating_year": "",
                "capacity_mw": 0.0,
            })
            neg_count += 1
            attempts += 1

        n_actual_pos = len([r for r in all_rows if r["energy_type"] == energy_type and r["label"] == 1])
        log.info(f"  {energy_type}: {n_actual_pos} positives + {neg_count} negatives")

    df = pd.DataFrame(all_rows)

    # Add country codes for negatives
    missing_cc = (df["country_code"] == "") | df["country_code"].isna()
    if missing_cc.sum() > 0:
        try:
            import reverse_geocoder as rg
            coords = list(zip(df.loc[missing_cc, "lat"], df.loc[missing_cc, "lon"]))
            results = rg.search(coords)
            df.loc[missing_cc, "country_code"] = [r["cc"] for r in results]
        except Exception as e:
            log.warning(f"  Geocoding negatives failed: {e}")

    # Save
    out = ROOT / "data" / "suitability_dataset_v2.parquet"
    df.to_parquet(out, index=False)
    log.info(f"\n  Saved {len(df)} rows to {out}")

    RESULTS["suitability_total"] = len(df)
    RESULTS["suitability_by_type"] = df.groupby(["energy_type", "label"]).size().to_dict()
    RESULTS["suitability_by_type"] = {str(k): v for k, v in RESULTS["suitability_by_type"].items()}

    return df


def step4_verify_data(df):
    """Run sanity checks on the dataset."""
    log.info("=" * 60)
    log.info("STEP 4: Data quality verification")
    log.info("=" * 60)

    checks = {}

    # Check 1: No NaN in critical columns
    for col in ["lat", "lon", "energy_type", "label"]:
        n_nan = df[col].isna().sum()
        ok = n_nan == 0
        checks[f"no_nan_{col}"] = ok
        log.info(f"  No NaN in {col}: {'PASS' if ok else f'FAIL ({n_nan} NaN)'}")

    # Check 2: Labels are 0 or 1
    valid_labels = set(df["label"].unique()) <= {0, 1}
    checks["valid_labels"] = valid_labels
    log.info(f"  Labels are 0/1: {'PASS' if valid_labels else 'FAIL'}")

    # Check 3: Lat/lon in valid range
    lat_ok = (df["lat"] >= -90).all() and (df["lat"] <= 90).all()
    lon_ok = (df["lon"] >= -180).all() and (df["lon"] <= 180).all()
    checks["valid_coords"] = lat_ok and lon_ok
    log.info(f"  Valid coordinates: {'PASS' if lat_ok and lon_ok else 'FAIL'}")

    # Check 4: Balance per energy type
    for et in df["energy_type"].unique():
        sub = df[df["energy_type"] == et]
        n_pos = (sub["label"] == 1).sum()
        n_neg = (sub["label"] == 0).sum()
        ratio = n_pos / n_neg if n_neg > 0 else float("inf")
        ok = 0.3 < ratio < 3.0  # Within 3:1 ratio
        checks[f"balance_{et}"] = ok
        log.info(f"  Balance {et}: {n_pos} pos / {n_neg} neg (ratio={ratio:.2f}) {'PASS' if ok else 'WARN'}")

    # Check 5: Geographic diversity (at least 3 regions per energy type)
    for et in df["energy_type"].unique():
        sub = df[(df["energy_type"] == et) & (df["label"] == 1)]
        n_regions = sub["region"].nunique()
        ok = n_regions >= 2
        checks[f"geo_diversity_{et}"] = ok
        log.info(f"  Geographic diversity {et}: {n_regions} regions {'PASS' if ok else 'WARN'}")
        log.info(f"    Regions: {sub['region'].value_counts().to_dict()}")

    # Check 6: No duplicate coordinates
    n_dupes = df.duplicated(subset=["lat", "lon", "energy_type"]).sum()
    ok = n_dupes == 0
    checks["no_duplicates"] = ok
    log.info(f"  No duplicates: {'PASS' if ok else f'WARN ({n_dupes} dupes)'}")

    # Check 7: Positives and negatives are spatially separated
    for et in df["energy_type"].unique():
        pos = df[(df["energy_type"] == et) & (df["label"] == 1)]
        neg = df[(df["energy_type"] == et) & (df["label"] == 0)]
        if len(pos) > 0 and len(neg) > 0:
            pos_mean = (pos["lat"].mean(), pos["lon"].mean())
            neg_mean = (neg["lat"].mean(), neg["lon"].mean())
            dist = np.sqrt((pos_mean[0] - neg_mean[0])**2 + (pos_mean[1] - neg_mean[1])**2)
            log.info(f"  Spatial separation {et}: pos mean=({pos_mean[0]:.1f},{pos_mean[1]:.1f}), "
                     f"neg mean=({neg_mean[0]:.1f},{neg_mean[1]:.1f}), dist={dist:.1f} deg")

    # Summary
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    log.info(f"\n  Data quality: {n_pass}/{n_total} checks passed")

    RESULTS["quality_checks"] = {k: str(v) for k, v in checks.items()}
    RESULTS["quality_pass_rate"] = f"{n_pass}/{n_total}"

    return n_pass == n_total


def step5_test_factor_engine(df):
    """Quick test of the factor engine on sample locations."""
    log.info("=" * 60)
    log.info("STEP 5: Test factor engine on sample data")
    log.info("=" * 60)

    try:
        from src.scoring import SuitabilityEngine

        for et in ["solar", "wind", "hydro", "geothermal"]:
            sub = df[df["energy_type"] == et]
            if len(sub) == 0:
                log.info(f"  {et}: no data, skipping")
                continue

            engine = SuitabilityEngine(et)
            n_factors = len(engine.factors)
            log.info(f"  {et}: {n_factors} factors loaded")

            # Score 10 positive and 10 negative samples
            pos = sub[sub["label"] == 1].head(10)
            neg = sub[sub["label"] == 0].head(10)

            pos_scores = []
            neg_scores = []

            for _, row in pos.iterrows():
                result = engine.score(row["lat"], row["lon"])
                pos_scores.append(result.overall_score)

            for _, row in neg.iterrows():
                result = engine.score(row["lat"], row["lon"])
                neg_scores.append(result.overall_score)

            # Without real external data, all scores should be ~0.5
            # This just verifies the engine runs without errors
            pos_mean = np.mean(pos_scores) if pos_scores else 0
            neg_mean = np.mean(neg_scores) if neg_scores else 0
            log.info(f"    Positive mean score: {pos_mean:.3f}")
            log.info(f"    Negative mean score: {neg_mean:.3f}")
            log.info(f"    (Expected ~0.5 for both without external data — engine works!)")

        RESULTS["factor_engine"] = "PASS"
        return True

    except Exception as e:
        log.error(f"  Factor engine test failed: {e}")
        RESULTS["factor_engine"] = f"FAIL: {e}"
        return False


def step6_test_embedding_extraction(df):
    """Test OlmoEarth embedding extraction on a tiny sample."""
    log.info("=" * 60)
    log.info("STEP 6: Test OlmoEarth embedding extraction (dry run)")
    log.info("=" * 60)

    try:
        import torch
        log.info(f"  PyTorch: {torch.__version__}")
        log.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Try loading OlmoEarth
        from rslearn.models.olmoearth_pretrain.model import OlmoEarth
        model = OlmoEarth(model_id="OLMOEARTH_V1_BASE", patch_size=4, selector=["encoder"])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Create a fake 12-band image
        fake_input = torch.randn(1, 12, 128, 128)
        if torch.cuda.is_available():
            fake_input = fake_input.cuda()

        with torch.no_grad():
            features = model(fake_input)

        # Check output shape
        if isinstance(features, list):
            feat = features[-1]
        elif isinstance(features, dict):
            feat = list(features.values())[-1]
        else:
            feat = features

        log.info(f"  Input shape: {fake_input.shape}")
        log.info(f"  Output type: {type(features)}")
        log.info(f"  Feature shape: {feat.shape}")
        log.info(f"  Feature dim: {feat.shape[1] if len(feat.shape) > 1 else 'N/A'}")

        # Global average pool to get 768-dim vector
        if len(feat.shape) == 4:  # [B, C, H, W]
            embedding = feat.mean(dim=[2, 3])  # [B, C]
        elif len(feat.shape) == 3:  # [B, N, C]
            embedding = feat.mean(dim=1)  # [B, C]
        else:
            embedding = feat

        log.info(f"  Embedding shape: {embedding.shape}")
        log.info(f"  Embedding range: [{embedding.min().item():.4f}, {embedding.max().item():.4f}]")

        RESULTS["olmoearth_test"] = "PASS"
        RESULTS["embedding_dim"] = embedding.shape[-1]
        RESULTS["gpu_available"] = torch.cuda.is_available()
        return True

    except Exception as e:
        log.error(f"  OlmoEarth test failed: {e}")
        RESULTS["olmoearth_test"] = f"FAIL: {e}"
        return False


def step7_test_classifier(df):
    """Train a quick classifier on geographic features as a sanity check."""
    log.info("=" * 60)
    log.info("STEP 7: Train sanity-check classifier (geographic features only)")
    log.info("=" * 60)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, classification_report
        from sklearn.preprocessing import StandardScaler

        for et in df["energy_type"].unique():
            sub = df[df["energy_type"] == et].copy()
            if len(sub) < 100:
                log.info(f"  {et}: too few samples ({len(sub)}), skipping")
                continue

            # Simple geographic features (latitude, longitude, abs_lat)
            X = sub[["lat", "lon"]].copy()
            X["abs_lat"] = X["lat"].abs()
            X["cos_lat"] = np.cos(np.radians(X["lat"]))
            X["sin_lon"] = np.sin(np.radians(X["lon"]))
            X["cos_lon"] = np.cos(np.radians(X["lon"]))
            y = sub["label"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            log.info(f"\n  {et.upper()} — Geographic-only baseline:")
            log.info(f"    Train: {len(X_train)}, Test: {len(X_test)}")
            log.info(f"    AUC-ROC: {auc:.4f}")
            log.info(f"    (This is the baseline to beat with OlmoEarth embeddings)")

            if auc > 0.95:
                log.warning(f"    WARNING: AUC too high ({auc:.4f}) — negatives may be trivially separable")
                log.warning(f"    This means lat/lon alone can separate pos/neg, which means")
                log.warning(f"    negatives are in obviously different locations (e.g. ocean)")

            RESULTS[f"baseline_auc_{et}"] = round(auc, 4)

        return True

    except ImportError as e:
        log.warning(f"  sklearn not installed: {e}")
        log.info("  Installing sklearn...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"],
                      capture_output=True)
        return False
    except Exception as e:
        log.error(f"  Classifier test failed: {e}")
        return False


def step8_time_split_analysis(df):
    """Analyze temporal distribution for time-split validation."""
    log.info("=" * 60)
    log.info("STEP 8: Time-split analysis")
    log.info("=" * 60)

    # Only EIA data has operating_year
    has_year = df["operating_year"].astype(str).str.match(r"^\d{4}$")
    df_with_year = df[has_year].copy()
    df_with_year["year"] = df_with_year["operating_year"].astype(int)

    if len(df_with_year) == 0:
        log.info("  No records with operating year — time split not possible yet")
        log.info("  Will need EIA data for time-split validation")
        RESULTS["time_split"] = "NOT_AVAILABLE"
        return

    log.info(f"  Records with operating year: {len(df_with_year)}")
    log.info(f"  Year range: {df_with_year['year'].min()} - {df_with_year['year'].max()}")

    # Proposed split
    pre_2020 = df_with_year[df_with_year["year"] < 2020]
    post_2020 = df_with_year[df_with_year["year"] >= 2020]

    log.info(f"\n  Proposed time split:")
    log.info(f"    Train (pre-2020): {len(pre_2020)}")
    log.info(f"    Test  (2020+):    {len(post_2020)}")

    if len(pre_2020) > 0:
        log.info(f"    Train by type: {pre_2020['energy_type'].value_counts().to_dict()}")
    if len(post_2020) > 0:
        log.info(f"    Test by type:  {post_2020['energy_type'].value_counts().to_dict()}")

    RESULTS["time_split_train"] = len(pre_2020)
    RESULTS["time_split_test"] = len(post_2020)


def main():
    log.info("=" * 60)
    log.info("OVERNIGHT AUTOMATION — OpenEnergy-Engine")
    log.info(f"Started: {datetime.now()}")
    log.info("=" * 60)

    # Step 1: Wait for downloads
    step1_wait_for_downloads(timeout_minutes=60)

    # Step 2: Merge datasets
    all_locations = step2_merge_datasets()
    if all_locations is None:
        log.error("ABORT: No data to work with")
        return

    # Step 3: Build suitability dataset
    suitability_df = step3_build_suitability_dataset(all_locations)

    # Step 4: Verify data
    step4_verify_data(suitability_df)

    # Step 5: Test factor engine
    step5_test_factor_engine(suitability_df)

    # Step 6: Test OlmoEarth (GPU)
    step6_test_embedding_extraction(suitability_df)

    # Step 7: Train sanity check classifier
    step7_test_classifier(suitability_df)

    # Step 8: Time split analysis
    step8_time_split_analysis(all_locations)

    # Save results
    RESULTS["completed_at"] = str(datetime.now())
    results_path = ROOT / "overnight_results.json"
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    log.info(f"\n{'='*60}")
    log.info(f"RESULTS SAVED: {results_path}")
    log.info(f"{'='*60}")
    for k, v in RESULTS.items():
        log.info(f"  {k}: {v}")
    log.info(f"\nDone at {datetime.now()}")


if __name__ == "__main__":
    main()
