# O3 EartH

**Geospatial Site Suitability Assessment Using Foundation Model Embeddings**

O3 EartH uses frozen [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) foundation model embeddings with lightweight classifiers to assess infrastructure site suitability across four energy types and six continents.

> Ziming Qi | Northeastern University

## Results

| Experiment | AUC-ROC |
|------------|---------|
| Geographic baseline (lat/lon) | 0.852 |
| OlmoEarth embeddings (768-dim) | 0.913 |
| **Combined (embeddings + geo)** | **0.927** |
| Spatial CV (63 countries) | 0.883 |
| Temporal (pre-2020 train, post-2020 test) | 0.952 |

### Per Energy Type (Combined, XGBoost)

| Type | AUC | Embedding Contribution |
|------|-----|----------------------|
| Solar | 0.975 | +6.2% over geo baseline |
| Wind | 0.924 | Policy-driven, geo stronger |
| Hydro | 0.922 | +10.1% over geo baseline |
| Geothermal | 0.948 | Subsurface-driven, geo stronger |

### Global Transfer

| Region | AUC |
|--------|-----|
| South America | 0.943 |
| Oceania | 0.923 |
| Africa | 0.919 |
| Europe | 0.912 |
| North America | 0.888 |
| Asia | 0.866 |

## How It Works

```
Sentinel-2 Image (12 bands, 10m)
    |
    v
OlmoEarth BASE (97M params, frozen)
    |
    v
768-dim Embedding (landscape features)
    |
    v
XGBoost Classifier --> Suitability Score (0-1)
```

1. **OlmoEarth** extracts 768-dimensional landscape representations from Sentinel-2 satellite imagery
2. **XGBoost** classifies patches as suitable/unsuitable for each energy type
3. **Factor Engine** provides configurable scoring across 19 factors (land cover, terrain, grid proximity, resource data, environmental constraints)

## Dataset

- **8,000 embeddings** extracted across solar, wind, hydro, geothermal
- **321,614 global energy locations** from EIA API + OSM Overpass
- **100+ countries** represented
- Available on HuggingFace: [2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth)

## Project Structure

```
O3earth/
├── THESIS.md                 # Full thesis document
├── src/
│   ├── factors/              # 19 configurable scoring factors
│   │   ├── solar.py          # GHI, cloud coverage, temperature
│   │   ├── wind.py           # Wind speed, direction, roughness
│   │   ├── hydro.py          # Water flow, elevation drop, watershed
│   │   ├── geothermal.py     # Heat flow, fault proximity, seismic
│   │   ├── infrastructure.py # Grid, roads, flood risk, protected areas
│   │   └── land.py           # Land cover, slope, elevation
│   ├── scoring/              # Suitability scoring engine
│   └── data_clients/         # EIA API + Planetary Computer clients
├── scripts/
│   ├── extract_embeddings.py # Sentinel-2 → OlmoEarth → embeddings
│   ├── train_suitability.py  # XGBoost/MLP training + ablation
│   ├── build_suitability_dataset.py
│   └── overnight_automation.py
├── data/                     # Parquet datasets
├── results/                  # Training results + predictions
└── archive/                  # Detection phase experiments (29 runs)
```

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost stackstac rioxarray

# Install OlmoEarth
pip install git+https://github.com/allenai/olmoearth_pretrain.git

# Extract embeddings (requires GPU + internet for Sentinel-2)
python scripts/extract_embeddings.py \
    --dataset data/suitability_dataset_v2_shuffled.parquet \
    --output-dir data/embeddings \
    --model-size BASE --resume

# Train classifier
python scripts/train_suitability.py \
    --embeddings data/embeddings/embeddings.npy \
    --metadata data/embeddings/embeddings_meta.csv \
    --output-dir results/suitability \
    --cv-folds 5
```

## Key References

- **OlmoEarth**: Allen Institute geospatial foundation model ([GitHub](https://github.com/allenai/olmoearth_pretrain))
- **TIML** (Tseng et al., 2022): Task-Informed Meta-Learning for Agriculture — methodological precedent
- **SatCLIP** (Klemmer et al., 2023): Location embeddings from satellite imagery

## License

MIT
