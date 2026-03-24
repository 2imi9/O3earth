---
tags:
  - renewable-energy
  - geospatial
  - satellite-imagery
  - site-suitability
  - olmoearth
  - sentinel-2
  - foundation-model
  - embeddings
license: mit
task_categories:
  - tabular-classification
language:
  - en
size_categories:
  - 10K<n<100K
---

# O3 EartH Dataset

Geospatial site suitability dataset with OlmoEarth foundation model embeddings for renewable energy infrastructure assessment.

**Key result:** OlmoEarth embeddings achieve AUC 0.867 under spatial cross-validation (leave-one-country-out).

## Files

| File | Description |
|------|-------------|
| suitability_dataset_v2_shuffled.parquet | 24,866 labeled samples (lat, lon, energy_type, label, country) |
| all_energy_locations.parquet | 321,614 global energy plant locations from EIA + OSM |
| embeddings/embeddings.npy | 8,000 OlmoEarth 768-dim embeddings |
| embeddings/embeddings_meta.csv | Metadata for each embedding |
| models/xgb_*.json | Trained XGBoost classifiers per energy type |
| models/scaler_*.pkl | StandardScaler for each energy type |

## How Embeddings Were Extracted

Sentinel-2 L2A (12 bands, 10m resolution) patches are passed through frozen OlmoEarth BASE encoder (97M params), then mean-pooled to produce a 768-dimensional landscape fingerprint per location.

- Source imagery: Microsoft Planetary Computer
- Model: allenai/olmoearth_pretrain (OLMOEARTH_V1_BASE)
- Patch size: 128x128 pixels (~1.28km)
- Time range: 2022-2023, max 30% cloud cover

## Coverage

- 100+ countries across 6 continents
- 4 energy types: solar (10K), wind (10K), hydro (4K), geothermal (866)
- Balanced positive (existing sites) and negative (random locations) samples

## Results

| Method | AUC |
|--------|-----|
| Geography only (lat/lon) | 0.579 |
| OlmoEarth embeddings | 0.902 |
| Spatial CV (leave-one-country-out) | 0.867 |

## Citation

Qi, Ziming. "O3 EartH: Geospatial Site Suitability Assessment Using Foundation Model Embeddings." 2026. Northeastern University.

## Links

- GitHub: https://github.com/2imi9/O3earth
- OlmoEarth: https://github.com/allenai/olmoearth_pretrain
