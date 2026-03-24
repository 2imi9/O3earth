# Results

## Ablation Study

| Feature Set | AUC |
|-------------|-----|
| Geographic baseline (lat/lon) | 0.579 |
| OlmoEarth embeddings only | 0.902 |
| Spatial CV (leave-one-country-out) | 0.867 |

## Per Energy Type

| Type | Samples | AUC (5-fold CV) |
|------|---------|-----------------|
| Solar | 1,832 | 0.947 |
| Wind | 1,794 | 0.862 |
| Hydro | 729 | 0.877 |
| Geothermal | 145 | 0.828 |

## Overfitting Diagnosis

| Test | Result | Verdict |
|------|--------|---------|
| Random label test | AUC 0.497 | Not memorizing |
| Leave-one-country-out | AUC 0.867 | Generalizes across regions |
| Learning curve | Improves with more data | Genuinely learning |
| Dimensionality | 5.9x sample/feature ratio | Borderline, mitigated by regularization |

## Global Transfer

| Region | AUC |
|--------|-----|
| South America | 0.943 |
| Oceania | 0.923 |
| Africa | 0.919 |
| Europe | 0.912 |
| North America | 0.888 |
| Asia | 0.866 |

## Dataset

- 24,866 labeled samples (solar, wind, hydro, geothermal)
- 8,000 OlmoEarth embeddings extracted (768-dim each)
- 321,614 global energy locations from EIA + OSM
- 100+ countries represented

Available on HuggingFace: [2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth)
