# Validation & Reproducibility

## Experimental Setup

- **Dataset:** 8,000 labeled samples across 212 countries, 4 energy types
- **Embeddings:** 768-dim vectors from frozen OlmoEarth BASE (97M params) on Sentinel-2 L2A
- **Classifiers:** XGBoost, Logistic Regression, MLP (all trained on same splits)
- **Negative sampling:** Random global locations matched by energy type count

## Ablation Study

Each feature set tested independently to isolate contributions.

| Feature Set | XGBoost AUC | What It Measures |
|-------------|-------------|-----------------|
| External features only (lat, lon, resource data) | 0.852 | Geography + known resource indicators |
| OlmoEarth embeddings only | 0.913 | Satellite-derived landscape representation |
| Combined (embeddings + external) | 0.927 | Full system |

**Finding:** OlmoEarth embeddings alone outperform external features by +6.1%, confirming that the foundation model captures suitability-relevant information beyond what traditional geospatial variables provide.

## Cross-Validation

5-fold stratified CV on the full dataset.

| Fold | AUC |
|------|-----|
| 1 | 0.911 |
| 2 | 0.898 |
| 3 | 0.894 |
| 4 | 0.917 |
| 5 | 0.934 |
| **Mean ± Std** | **0.911 ± 0.015** |

Low variance across folds indicates stable, reproducible results.

## Spatial Cross-Validation (Leave-One-Country-Out)

Standard CV can leak geographic information when nearby locations appear in both train and test. Leave-one-country-out CV eliminates this by holding out entire countries.

- **Countries evaluated:** 63 (each with ≥5 positive and ≥5 negative samples)
- **AUC: 0.867 ± 0.114**

The 4.4% drop from standard CV (0.911 → 0.867) quantifies the geographic leakage present in random splits. **0.867 is the conservative, defensible result.**

## Temporal Validation

Train on locations where energy plants were built before 2020. Test on plants built 2020–2025.

- **Train:** 291 pre-2020 positives + 1,854 negatives
- **Test:** 126 post-2020 positives + 796 negatives

| Method | AUC | Interpretation |
|--------|-----|----------------|
| Standard temporal split | 0.877 | Partially inflated by spatial proximity |
| Distance-only baseline | 0.818 | New plants built near existing ones |
| **Leakage-controlled (>200km)** | **0.852** | **Honest temporal signal** |

### Spatial Autocorrelation Analysis

Post-2020 plants average 262km from the nearest pre-2020 plant, while negatives average 929km. This proximity inflates the standard temporal AUC. Evaluating only on post-2020 plants >200km from any training plant yields AUC=0.852 — still above the distance-only baseline (0.818) by +3.4%, confirming the model captures genuine suitability signals beyond spatial proximity.

## Regional Generalization

AUC computed per continent to assess geographic transfer.

| Region | AUC | Sample Count |
|--------|-----|-------------|
| South America | 0.943 | varies |
| Oceania | 0.923 | varies |
| Africa | 0.919 | varies |
| Europe | 0.912 | varies |
| North America | 0.888 | varies |
| Asia | 0.866 | varies |

Performance is consistent across continents. Lower Asian AUC may reflect greater landscape diversity or policy-driven siting patterns in that region.

## Sanity Checks

| Test | Expected | Observed | Pass |
|------|----------|----------|------|
| Random label shuffle | ~0.50 | 0.497 | ✅ |
| Zero NaN in embeddings | 0 | 0 | ✅ |
| Zero degenerate rows | 0 | 0 | ✅ |
| All 768 dims active (std > 0.001) | 768 | 768 | ✅ |
| Learning curve improves with data | monotonic | yes | ✅ |

## Overfitting Diagnosis

| Check | Result | Assessment |
|-------|--------|-----------|
| Feature-to-sample ratio | 768 features / 8,000 samples = 10.4x | Acceptable |
| Random label AUC | 0.497 | Model does not memorize noise |
| Learning curve | 0.82 (10%) → 0.90 (100%) | Improves with data, not memorizing |
| Spatial CV drop | 0.911 → 0.867 (−4.4%) | Some geographic signal, not pure overfitting |
| Dimensionality | PCA top 50 dims explain 97.3% variance | Effective dimensionality is moderate |

## Comparison to Related Work

| Work | Task | Evaluation Samples | Our Comparison |
|------|------|--------------------|---------------|
| TIML (Tseng et al., ICML) | Crop classification | 26–1,345 per task | 8,000 samples |
| SatCLIP (Klemmer, NeurIPS 2023) | Geo embeddings | ~5,000 downstream | 8,000 samples |
| Jean et al. (Science 2016) | Poverty prediction | ~4,000 villages | 8,000 samples |
| You et al. (AAAI 2017) | Crop yield | ~3,000 counties | 8,000 samples |

Our dataset size, geographic diversity (212 countries), and multi-method validation (spatial CV, temporal, regional) meet or exceed standards in comparable published work.

## Reproducibility

All data, embeddings, trained models, and code are publicly available:

- **Code:** [github.com/2imi9/O3earth](https://github.com/2imi9/O3earth)
- **Data + Models:** [huggingface.co/datasets/2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth)

To reproduce results:
```bash
git clone https://github.com/2imi9/O3earth.git
cd O3earth
pip install -r requirements.txt
python scripts/train_suitability.py
```
