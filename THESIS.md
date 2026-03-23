# OpenEnergy-Engine: Geospatial Site Suitability Assessment Using Foundation Model Embeddings — A Renewable Energy Case Study

**Author:** Ziming (Frank) Qi
**Institution:** Northeastern University
**Program:** Millennium Fellowship Research
**Mentor:** Professor Auroop Ratan Ganguly
**Repository:** github.com/2imi9/OpenEnergy-Engine

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Research Gap & Contribution](#2-research-gap--contribution)
3. [Background & Literature Review](#3-background--literature-review)
4. [System Architecture](#4-system-architecture)
5. [OlmoEarth Foundation Model](#5-olmoearth-foundation-model)
6. [Factor Engine Design](#6-factor-engine-design)
7. [Dataset Construction](#7-dataset-construction)
8. [Validation & Evaluation Framework](#8-validation--evaluation-framework)
9. [Experimental Results](#9-experimental-results)
10. [Platform Architecture](#10-platform-architecture)
11. [Datasets & Data Sources](#11-datasets--data-sources)
12. [Infrastructure & Stack](#12-infrastructure--stack)
13. [Limitations & Generalizability](#13-limitations--generalizability)
14. [NEMS Valuation (Future Work)](#14-nems-valuation-future-work)
15. [References](#15-references)
16. [Appendix A: Mentor Communication](#appendix-a-mentor-communication)
17. [Appendix B: Detection Phase (Lessons Learned)](#appendix-b-detection-phase-lessons-learned)

---

## 1. Abstract

We present a generalizable framework for geospatial site suitability assessment using foundation model embeddings, demonstrated through the application of renewable energy facility siting across four energy types (solar, wind, hydro, geothermal) and six continents. Our approach uses frozen OlmoEarth encoder embeddings (768-dimensional landscape representations from Sentinel-2 imagery) as features for lightweight statistical classifiers (XGBoost), following the representation learning paradigm established by Tseng et al. (2022) for agricultural applications. Unlike existing approaches that rely on either classical GIS multi-criteria decision analysis (MCDA) or deep learning for infrastructure detection alone, our system uses foundation model embeddings as a learned land characterization layer within a broader configurable suitability scoring pipeline. We validate through a multi-pronged framework: ablation studies show embeddings improve AUC from 0.852 (geographic baseline) to 0.927 (combined); leave-one-country-out spatial cross-validation across 63 countries yields AUC=0.883; temporal validation (training on pre-2020 plants, testing on 2020-2025 plants) achieves AUC=0.952; and the model transfers strongly to data-scarce regions including Africa (AUC=0.919) and South America (AUC=0.943). Per-energy-type analysis reveals that embeddings contribute most where visual land characteristics drive suitability (solar: +6.2%, hydro: +10.1%), and less where subsurface or policy factors dominate (wind, geothermal). While demonstrated on renewable energy, the methodology generalizes to any location-dependent infrastructure siting problem (e.g., EV charging, battery storage, data centers, agricultural facilities). Dataset and embeddings are publicly available at huggingface.co/datasets/2imi9/O3earth.

---

## 2. Research Gap & Contribution

### The Gap

No published work combines a geospatial foundation model with configurable multi-criteria suitability scoring for infrastructure site recommendation. Current approaches fall into three categories:

| Approach | Examples | Limitation |
|----------|---------|------------|
| **Classical GIS-MCDA** | AHP/TOPSIS overlay studies | No deep learning, requires manual GIS expertise, not scalable |
| **DL for detection** | DeepSolar, Global Renewables Watch | Answers "what's there" not "should you build here?" |
| **Resource atlases** | Global Solar Atlas, Global Wind Atlas, RE Explorer | Static maps, no ML, no risk integration, no economics |

### Our Contributions

1. **Generalizable framework:** First system combining geospatial foundation model embeddings with configurable multi-factor suitability scoring for infrastructure site recommendation
2. **Representation learning for suitability:** Demonstrating that frozen foundation model embeddings + lightweight classifiers (XGBoost/MLP) outperform traditional GIS features — following the paradigm from Tseng et al. (2022) but applied to energy infrastructure
3. **Multi-pronged validation:** Retroactive temporal prediction, cross-region transfer, expert-alignment with NREL/IRENA, negative validation, and SHAP interpretability — addressing the absence of existing benchmarks
4. **Cross-region transfer:** Train on data-rich regions (US/EU), predict on data-scarce regions (developing nations)
5. **Open pipeline:** Fully reproducible from data ingestion to prediction, built on open models and data
6. **Broader applicability:** While demonstrated on renewable energy, the methodology applies to any location-dependent infrastructure siting problem

### Why This Matters

> "We demonstrate that pretrained geospatial foundation model embeddings, combined with lightweight statistical classifiers, achieve strong suitability prediction (AUC=0.927 combined, 0.883 spatial CV across 63 countries) without the computational cost of end-to-end fine-tuning — making the system deployable at scale for infrastructure site recommendation. Temporal validation (AUC=0.952) confirms the model predicts future development sites from historical patterns."

### Methodological Justification

A key design decision is using frozen OlmoEarth embeddings with statistical classifiers rather than end-to-end fine-tuning. This follows the representation learning paradigm validated across domains:

| Precedent | Approach | Result |
|-----------|----------|--------|
| **TIML (Tseng et al., 2022)** | Satellite time series → LSTM encoder → meta-learned classifier | SOTA on agricultural classification in data-sparse regions |
| **SatCLIP (Klemmer et al., 2023)** | Location + satellite embeddings → linear classifier | +8-12% over location-only baselines |
| **CLIP (Radford et al., 2021)** | Frozen image embeddings → linear probe | Competitive with fine-tuned models |
| **Ours** | Sentinel-2 → frozen OlmoEarth encoder → XGBoost | AUC=0.927 (combined), 0.883 (spatial CV, 63 countries) |

The intelligence resides in the **representation** (768-dim embeddings that capture land cover, terrain texture, vegetation, infrastructure patterns), not in the final classifier. This design enables:
- **Scalability:** Extract once, predict anywhere — no GPU needed at inference for pre-computed grids
- **Interpretability:** SHAP values on XGBoost features reveal which landscape characteristics drive suitability
- **Modularity:** Easy to add/remove factors without retraining the foundation model
- **Generalizability:** Same embeddings can serve multiple downstream tasks (solar siting, wind siting, risk assessment, etc.)

---

## 3. Background & Literature Review

### 3.1 Renewable Energy Site Selection Methods

**GIS-MCDA (dominant approach):**
- Weighted overlay of rasterized criteria layers using AHP, TOPSIS, or ELECTRE
- Requires domain expert to define and weight criteria manually
- Examples: Multi-criteria GIS for Solar & Wind (ScienceDirect, 2025); Smart GIS MCDM for Solar-Wind Hybrid (Springer, 2025)

**Deep learning for detection:**
- U-Net, DeepLabv3+, FCN with ResNet backbones
- Global Renewables Watch (arXiv 2503.14860, 2025): 375,197 wind turbines + 86,410 solar farms detected from 13 trillion pixels. Detection, not suitability.
- DeepSolar: 93% precision at 0.3m resolution for rooftop solar. Not applicable at 10m.
- Deep Learning Ensemble for Solar Potential Mapping (Springer, 2025): F1=0.91, but focused on rooftop potential, not utility-scale suitability.

**Explainable AI approaches:**
- Global Spatial Suitability Mapping with XAI (MDPI IJGI, 2022): Random Forest / SVM / MLP trained on 55,000+ real plant locations with SHAP explanations. Closest to our approach but no foundation model, no configurable factors, no economic integration.

**Hybrid models:**
- China wind-solar-hydrogen site selection (ScienceDirect, 2025): Stage 1 GIS macro-screening, Stage 2 MCDM micro-ranking.
- Spain hybrid renewable systems (Frontiers, 2025): Multi-criteria combined with grid capacity analysis.

### 3.2 Representation Learning & Transfer from Foundation Models

A growing body of work demonstrates that frozen pretrained encoders combined with lightweight classifiers match or outperform end-to-end fine-tuned models, especially in data-scarce settings:

- **TIML (Tseng, Kerner, Rolnick, 2022):** Task-Informed Meta-Learning for agriculture. Uses satellite time series → LSTM encoder → meta-learned classifier. Achieves SOTA on crop type classification in Kenya, Brazil, Togo. Key insight: spatially-defined task metadata (geographic coordinates, crop category) improves cross-region transfer. *Directly validates our approach architecture.*
- **SatCLIP (Klemmer et al., 2023):** Location encoder trained on satellite imagery. Frozen embeddings + linear classifiers outperform location-only baselines by 8-12% on geospatial prediction tasks.
- **SSL4EO (Wang et al., 2022):** Self-supervised pretraining on satellite imagery boosts downstream land use classification by 10-20% over random initialization.
- **CLIP (Radford et al., 2021):** Demonstrated that frozen image embeddings + linear probes achieve competitive performance across dozens of tasks without fine-tuning. Established the "representation learning" paradigm we follow.

**Our position:** We extend this paradigm to infrastructure site suitability — using OlmoEarth (a geospatial foundation model) as the frozen encoder and XGBoost/MLP as the downstream classifier. The novelty is in the application domain (energy siting) and the multi-factor integration, not in the representation learning mechanism itself.

### 3.3 Geospatial Foundation Models

| Model | Developer | Params | Key Strength |
|-------|-----------|--------|--------------|
| **OlmoEarth** | Allen AI | Nano to Large | SOTA on segmentation/classification/regression; open; S1+S2; marine infra detection |
| **Prithvi-EO-2.0** | NASA/IBM | 600M | 4.2M global time series, temporal + location embeddings, cloud gap imputation |
| **Clay** | Development Seed | Open | Fully open (incl. training data), similarity search for solar panels |
| **SatMAE** | Meta | -- | Masked autoencoder on satellite imagery |

**Critical gap:** No published work uses these models for end-to-end suitability scoring. Applications are limited to detection, segmentation, and environmental monitoring.

### 3.4 Risk Analytics for Renewable Energy

- **Physical climate risk:** Extreme weather increasing 20-30% by 2050 (TUV SUD, 2025). Droughts disrupting hydropower; hurricanes/wildfires threatening solar/wind.
- **Grid/interconnection risk:** 90% of US grid queue projects are renewable but face congestion and delayed interconnections (WEF Global Energy, 2026).
- **Economic/financial risk:** Asset-level climate-related financial risk assessment considering physical + transition risks (ScienceDirect). NBER general equilibrium model for transition risk.
- **IRENA Adaptation Framework (April 2025):** Metrics for renewable energy as climate risk mitigator.

### 3.5 Existing Platforms

| Platform | Provider | What It Does | What It Lacks |
|----------|----------|-------------|---------------|
| Global Solar Atlas | World Bank / Solargis | GHI/DNI maps, PV yield calculator | No ML, no land analysis, solar only |
| Global Wind Atlas | World Bank / DTU | Wind speed/power density | No ML, no suitability scoring |
| IRENA Global Atlas | IRENA | 700+ renewable maps | View-only, no scoring engine |
| Solargis | Solargis s.r.o. | Bankable solar resource data | Commercial, closed, solar only |
| RE Explorer | NREL | Resource maps + exclusion layers | No ML, US-focused |
| NEMS | EIA | LCOE/LACE modeling | No spatial/geospatial component |

---

## 4. System Architecture

```
                    +------------------+
                    | Sentinel-2 Patch |
                    | (12-band, 10m)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | OlmoEarth ViT    |
                    | (feature extractor)|
                    | 768-dim embeddings|
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
   +----------v----------+     +-----------v-----------+
   | Landscape Features  |     | External Data APIs    |
   | (land cover, terrain|     | (GHI, wind speed,     |
   |  vegetation, water, |     |  elevation, grid      |
   |  built-up density)  |     |  distance, flood risk)|
   +----------+----------+     +-----------+-----------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v---------+
                   | Factor Engine     |
                   | (configurable per |
                   |  energy type)     |
                   +--------+---------+
                            |
                   +--------v---------+
                   | Suitability Score |
                   | (0-1 per energy   |
                   |  type)            |
                   +--------+---------+
                            |
                   +--------v---------+
                   | NEMS Valuation    |
                   | (NPV/IRR/LCOE)   |  <- Future work
                   +------------------+
```

**Key insight:** OlmoEarth is ONE input to the factor engine, not the entire system. It handles "what does this land look like?" while external APIs handle domain-specific questions (how much sun, how much wind, how far from grid).

---

## 5. OlmoEarth Foundation Model

### How It Works (Analogy to LLMs)

| | LLM (Language) | OlmoEarth (Satellite) |
|---|---|---|
| **Input** | Text tokens | Pixel patches (12-band, 4x4 blocks) |
| **Tokenization** | Words to embeddings | Patches to embeddings |
| **Encoder** | Learns semantic meaning of words | Learns semantic meaning of land |
| **Latent space** | 768-dim vector per token | 768-dim vector per patch |
| **Decoder** | Generates answer | Generates classification / feature map |

### Input
- 12-band Sentinel-2 image: `[batch, 12, H, W]`
- Bands: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
- Normalized: per-band `(val - (mean - 2*std)) / (4*std)`
- Each pixel = 10m x 10m; a 128x128 patch covers 1.28km x 1.28km

### Output (as feature extractor)
- Feature maps: `[batch, 768, H/4, W/4]` for BASE model (~97M params)
- 768-dim features per spatial position
- Each position represents 40m x 40m area
- Features encode: land cover type, vegetation density, built-up area, water bodies, terrain texture

### Model Sizes
| Size | Use Case |
|------|----------|
| OLMOEARTH_V1_NANO | Debugging |
| OLMOEARTH_V1_TINY | Lightweight |
| OLMOEARTH_V1_BASE | Production (what we use, 97M params) |
| OLMOEARTH_V1_LARGE | Best quality (needs A100+) |

### Why OlmoEarth for Suitability (Not Detection)

OlmoEarth at 10m resolution cannot competitively detect individual solar panels (~2m) or wind turbines. But it excels at landscape-scale understanding:

| What OlmoEarth captures | Why it matters for suitability |
|---|---|
| Land cover gradients | Not just "is cropland" but "how much, what condition" |
| Terrain texture | Slope, roughness, drainage patterns |
| Vegetation density | Environmental constraints, agricultural value |
| Built-up proximity | Infrastructure access, demand centers |
| Water bodies | Hydro potential, flood risk |
| Temporal changes | Construction activity, land use transitions |

Static GIS datasets (NLCD) provide 16 categorical classes updated every 5 years. OlmoEarth provides 768-dimensional continuous features from current imagery that work globally without region-specific training data.

---

## 6. Factor Engine Design

### Configurable Factors Per Energy Type

Each factor can be toggled on/off and weighted per energy type:

#### General Factors (always on)
| Factor | Data Source | What It Measures |
|--------|-----------|------------------|
| Land cover / land use | OlmoEarth embeddings + NLCD | Terrain suitability |
| Terrain (slope, aspect, elevation) | USGS 3DEP / SRTM | Physical buildability |
| Grid proximity | EIA / OpenStreetMap | Transmission access |
| Road access | OpenStreetMap | Construction logistics |
| Environmental constraints | EPA / protected areas DB | Regulatory exclusions |
| Flood risk | FEMA / global flood models | Physical risk |

#### Solar-Specific
| Factor | Data Source |
|--------|-----------|
| GHI / DNI (solar irradiance) | NREL NSRDB / Global Solar Atlas |
| Cloud coverage frequency | ERA5 reanalysis / Sentinel-2 cloud masks |
| Precipitation patterns | GPM / CHIRPS |
| Temperature effects on efficiency | ERA5 / weather stations |

#### Wind-Specific
| Factor | Data Source |
|--------|-----------|
| Wind velocity and direction | NREL Wind Toolkit / Global Wind Atlas |
| Turbulence intensity | ERA5 reanalysis |
| Terrain roughness | SRTM DEM derived |
| Wake effects | Modeled from existing turbine locations |

#### Hydro-Specific
| Factor | Data Source |
|--------|-----------|
| Water source stability | HydroSHEDS / GRDC |
| Water flow speed | GRDC river discharge |
| Elevation drop / head | SRTM DEM |
| Watershed health | Sentinel-2 vegetation indices |

#### Geothermal-Specific
| Factor | Data Source |
|--------|-----------|
| Heat flow density | USGS heat flow database |
| Lithology / rock type | National geological surveys |
| Fault proximity | USGS fault database |
| Seismic activity | USGS earthquake catalog |

---

## 7. Dataset Construction

### Task Reframing

| | Old (Detection) | New (Suitability) |
|---|---|---|
| **Question** | "Is this pixel a solar panel?" | "Is this 1.28km2 area suitable for solar?" |
| **Label** | Per-pixel class (0/1/2) | Per-patch binary (suitable / not suitable) |
| **Output** | Segmentation mask | Suitability score 0-1 |
| **Difficulty** | Very hard at 10m | Easier -- landscape-scale |

### Global Training Data

**Positive examples** = locations where plants actually exist (someone decided this site is suitable)
**Negative examples** = locations with no energy infrastructure (smart sampling)

| Dataset Component | Source | Count Target |
|---|---|---|
| Solar positives | OSM polygons (1.66M globally) | ~5,000 patches |
| Solar negatives | Random global, no nearby energy infra | ~5,000 patches |
| Wind positives | OSM + Global Renewables Watch | ~3,000 patches |
| Wind negatives | Random global | ~3,000 patches |
| Hydro positives | OSM dams + GRanD database | ~2,000 patches |
| Hydro negatives | Random | ~2,000 patches |

**Smart negative sampling:** Not random ocean/desert. Select plausible-looking locations (flat land, near roads) that DON'T have plants. This forces the model to learn real suitability signals rather than trivial distinctions.

### Time-Split for Validation

```
Training set:   Plants built before 2020 (from EIA 860 + OSM)
Validation set: Plants built 2020-2022
Test set:       Plants built 2023-2025
Question:       Does the model score NEW plant locations highly?
```

### Geographic Split for Transfer

```
Training regions:    US + Europe (data-rich, well-labeled)
Transfer test:       India, Brazil, Sub-Saharan Africa (data-scarce)
Question:            Does the model generalize globally?
```

---

## 8. Validation & Evaluation Framework

### Why We Define Our Own Benchmark

This is a novel task — no existing benchmark exists for foundation-model-based suitability scoring. We define our own evaluation rubric and must demonstrate its validity through multiple independent lines of evidence. A single metric is insufficient; the combination of five validation methods provides credibility.

### Test 1: Retroactive Temporal Prediction (Strongest Proof)

- Train on plants built **before 2020** (from EIA 860 operating_year + OSM)
- Test: does the model score high on locations where plants were built **2021-2025**?
- If yes → the model predicted future development it never saw
- **Metric:** % of new plants in top 20% scored areas; AUC on temporal test set
- **Precedent:** Tseng et al. (2022) use identical temporal validation for yield estimation (Table 2)

### Test 2: Cross-Region Transfer

- Train on US + Europe (data-rich, well-labeled)
- Score sites in India, Brazil, Kenya, Australia (data-scarce)
- Different policies, different economics, same physics
- If the model works → it learned land suitability, not policy patterns
- **Metric:** AUC-ROC on held-out regions

### Test 3: Expert Alignment (External Validation)

- Compare model's top-scored undeveloped sites against:
  - NREL technical potential maps (solar/wind resource zones)
  - IRENA renewable energy zones
  - Government-designated renewable energy corridors
- **Metric:** Spatial overlap (%) between model-identified high-suitability areas and expert-designated zones

### Test 4: Negative Validation (Sanity Check)

- Score locations everyone agrees are unsuitable:
  - Dense urban centers (Manhattan, downtown Tokyo)
  - Protected national parks
  - Ocean / deep water
  - Steep mountains (for solar)
- Model must score these low
- **Metric:** Mean score of known-bad locations vs. known-good

### Test 5: Ablation Study (Component Contribution)

| Model Variant | Features | AUC-ROC |
|---|---|---|
| Random baseline | Random scores | 0.50 |
| Geographic only | Lat/lon features (8 dims) | **0.852** |
| OlmoEarth only | 768-dim embeddings | **0.913** |
| **Combined** | **Embeddings + Geographic** | **0.927** |
| Spatial CV (63 countries) | Leave-one-country-out | **0.883** |
| Temporal validation | Pre-2020 → post-2020 | **0.952** |

**Key result:** OlmoEarth embeddings improve AUC from 0.852 → 0.913 (+6.1%) over geographic baseline. Combined features achieve 0.927. Temporal validation (0.952) confirms genuine predictive capability.

### Interpretability (SHAP Analysis)

- Run SHAP on XGBoost to identify which of the 768 embedding dimensions drive predictions
- Map top dimensions back to landscape characteristics (e.g., "dim 25 correlates with flat open terrain")
- If reasons are physically meaningful → model has learned real suitability signals
- If reasons are spurious (e.g., "scored high because latitude = 35°") → model may be overfitting to geography

### Metrics

| Metric | What It Measures | Why |
|--------|-----------------|-----|
| **AUC-ROC** | Separation of suitable vs unsuitable | Main metric -- "can the model rank sites?" |
| **Precision@K** | Of top K sites, how many are real? | Practical value |
| **Recall@K** | Of all real sites, how many in top K? | Coverage |
| **Calibration** | Score 0.8 = ~80% actual plants? | Trust |
| **Ablation AUC delta** | With vs without OlmoEarth | Thesis proof |
| **Cross-region AUC** | Train US/EU, test elsewhere | Global proof |

---

## 9. Experimental Results

### Phase 1: Detection Experiments (29 runs) — Completed

These experiments established that OlmoEarth + Sentinel-2 at 10m resolution is better suited for landscape characterization than sub-pixel infrastructure detection.

| Run | Config | Key Change | Best F1 | Outcome |
|-----|--------|-----------|---------|---------|
| 1-18 | Various | EIA circle labels, unbalanced data | 0.00 | Model predicted all-background |
| 19 | v8_balanced | Balanced dataset, EIA labels | 0.33 | Labels too noisy at 10m |
| 20 | v9_osm | OSM polygon labels, unfrozen encoder | 0.32 | Encoder collapse |
| 21 | v10_frozen | Frozen encoder + OSM labels | **0.40** | First real detection |
| 22 | v11_fullfreeze | Fully frozen encoder | **0.46** | Best detection result |
| 23-29 | Various | Architecture/LR/loss variations | 0.00-0.46 | No improvement beyond 0.46 |

**Key findings:**
1. Dataset imbalance (94% bg-only) was the primary failure mode for runs 1-18
2. EIA centroid labels (500m buffer circles) have no spectral contrast at 10m — B08 ratio ~0.97 between "solar" and "background"
3. OSM polygon labels show clear spectral contrast — B08 ratio 0.72
4. Frozen encoder essential — unfreezing always caused collapse
5. **F1=0.46 ceiling** driven by 10m resolution limitation + only 340 solar training windows

**Insight that drove the pivot:** OlmoEarth's 768-dim features clearly encode landscape-level information (run 21/22 prove the encoder learns useful representations). The limitation is in the decoder task (pixel segmentation at 10m), not the encoder. Using the encoder as a feature extractor for suitability scoring leverages its strength.

### Phase 2: Suitability Scoring — Results

#### Dataset Statistics
| Component | Count | Source |
|-----------|-------|--------|
| Full dataset | 24,866 | EIA 860 + OSM Overpass (global) |
| Embeddings extracted | 8,000 | Shuffled across all types |
| Solar (pos + neg) | 3,207 (2,145 + 1,062) | 100+ countries |
| Wind (pos + neg) | 3,248 (2,180 + 1,068) | 100+ countries |
| Hydro (pos + neg) | 1,285 (866 + 419) | 60+ countries |
| Geothermal (pos + neg) | 260 (159 + 101) | 20+ countries |
| **HuggingFace:** | [2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth) | Public dataset |

#### Ablation Study (XGBoost, 8,000 samples, all energy types)

| Feature Set | AUC-ROC | Avg Precision | Delta vs Geo |
|-------------|---------|---------------|-------------|
| Geographic features only (lat/lon) | 0.852 | 0.902 | — |
| **OlmoEarth embeddings only (768-dim)** | **0.913** | **0.947** | **+6.1%** |
| **Embeddings + Geographic combined** | **0.927** | **0.957** | **+7.5%** |

5-fold cross-validation on embeddings: **AUC = 0.911 ± 0.015** (stable, low variance)

#### Per-Energy-Type Ablation (XGBoost)

| Energy Type | N | Geo Only | OlmoEarth Only | Combined | Embedding Δ |
|-------------|---|----------|----------------|----------|-------------|
| Solar | 3,207 | 0.901 | 0.963 | **0.975** | **+6.2%** |
| Wind | 3,248 | 0.902 | 0.888 | **0.924** | -1.4% |
| Hydro | 1,285 | 0.817 | 0.918 | **0.922** | **+10.1%** |
| Geothermal | 260 | 0.938 | 0.870 | **0.948** | -6.8% |

**Key finding:** Embeddings help most where **visual land characteristics** determine suitability (solar: flat open land, hydro: water features). For energy types driven by subsurface or policy factors (wind: subsidies/zoning, geothermal: underground heat flow), geographic features are stronger — but combined always wins. This confirms the model captures physical site characteristics rather than policy patterns.

#### Spatial Cross-Validation (Leave-One-Country-Out)

**Mean AUC = 0.883 ± 0.093 across 63 countries** (the honest, spatially rigorous number)

| Region | AUC | N |
|--------|-----|---|
| South America | **0.943** | 170 |
| Oceania | **0.923** | 198 |
| Africa | **0.919** | 186 |
| Europe | **0.912** | 343 |
| North America | **0.888** | 374 |
| Asia | **0.866** | 329 |

The model generalizes globally. Africa (0.919) and South America (0.943) perform as well or better than North America — demonstrating the model is not simply memorizing US/EU development patterns but learning transferable geospatial representations.

**Bottom performers:** China (0.621), Kazakhstan (0.599) — vast countries with extremely diverse land types, where a single country-level holdout loses significant training signal.

#### Temporal Validation (Strongest Evidence)

**Train on pre-2020 plants → Test on 2020-2025 plants**

| Model | AUC |
|-------|-----|
| Logistic Regression | **0.952** |
| XGBoost | **0.952** |

The model trained on historical plant locations correctly predicts where **new** plants were subsequently built. This is the strongest evidence that the model captures genuine suitability signals rather than memorizing existing development patterns.

#### SHAP Interpretability Analysis

Top 5 most important embedding dimensions: dim_612, dim_748, dim_559, dim_626, dim_370

**Signal concentration:**
- 113 dims capture 50% of the predictive signal
- 354 dims capture 80%
- 496 dims capture 90%

The signal is distributed across the embedding space — the model leverages diverse visual features (land cover, terrain texture, vegetation patterns, water bodies, built-up area indicators) rather than relying on a small number of dimensions. This is consistent with the broad landscape characterization learned during OlmoEarth's pretraining.

#### Overfitting Diagnostics

| Test | Result | Verdict |
|------|--------|---------|
| Random label test | AUC = 0.497 | ✅ Not memorizing noise |
| Leave-one-country-out | AUC = 0.883 | ✅ Generalizes spatially |
| 5-fold CV stability | 0.911 ± 0.015 | ✅ Low variance |
| Temporal validation | AUC = 0.952 | ✅ Predicts future development |
| Learning curve | 0.82→0.90 (scales with data) | ✅ Not saturated |

#### Comparison to Related Work

| Paper | Task | Samples | AUC | Our Result |
|-------|------|---------|-----|------------|
| TIML (Tseng 2022) | Crop classification | 306-1,345/task | 0.890 mean | **0.927** |
| SatCLIP (Klemmer 2023) | Geo embeddings | ~5K downstream | +8-12% over baseline | **+7.5%** |
| Jean et al. 2016 | Poverty prediction | ~4,000 | N/A | — |
| Global Suitability XAI (MDPI 2022) | Energy suitability | 55,000+ | ~0.85 | **0.927** |
| **Ours** | **Site suitability** | **8,000** | **0.927** | — |

#### Remaining Experiments
- [ ] Expert alignment with NREL/IRENA technical potential zones
- [ ] Partial fine-tuning comparison (unfreeze last 2-3 OlmoEarth layers)
- [ ] Resume extraction to 24,866 samples (optional — 8,000 sufficient)
- [ ] Per-energy-type SHAP analysis

### Best Checkpoint

- **Path:** `project_data/openenergyengine/run_21_frozen/best.ckpt`
- **Model:** OlmoEarth BASE, frozen encoder, UNet decoder
- **Use:** Feature extractor for land characterization embeddings

---

## 10. Platform Architecture

The project consists of two repositories:

### Model/Engine Repo (FT_olmoearth — this repo)
- OlmoEarth fine-tuning and feature extraction
- Factor engine implementation
- Suitability scoring pipeline
- Dataset construction and validation

### Application Platform (2imi9/OpenEnergy-Engine)
Already built, waiting for the suitability model:

| Component | Technology | Status |
|---|---|---|
| OlmoEarth Model | ViT backbone, multi-task heads | Architecture complete |
| Climate Risk Model | ACE2-inspired, SSP projections | Architecture complete |
| NEMS Valuation Engine | NPV/IRR/LCOE calculations | Fully functional |
| EIA Data Pipeline | EIA API v2 (860/923/AEO) | Fully functional |
| Satellite Data Pipeline | Planetary Computer STAC API | Fully functional |
| LLM Integration | vLLM (Qwen3-8B) or NVIDIA NIM | Fully functional |
| MCP Server | Model Context Protocol | Fully functional |
| REST API | FastAPI with Swagger docs | Fully functional |
| Web Dashboard | Streamlit, interactive maps, AI chat | Fully functional |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /api/health | GET | Module availability check |
| /api/detect | POST | Satellite-based detection |
| /api/climate-risk | POST | Climate risk assessment |
| /api/value-asset | POST | 25-year NPV/IRR/LCOE valuation |
| /api/eia/generators | GET | EIA generator inventory |
| /api/eia/generation/{state} | GET | State generation data |
| /api/eia/prices | GET | Price forecasts |
| /api/eia/capacity/{source} | GET | Capacity forecasts |
| /api/analyze | POST | LLM-powered analysis |
| /api/report | POST | Formatted report generation |

---

## 11. Datasets & Data Sources

| Dataset | Source | Description | Coverage |
|---|---|---|---|
| Sentinel-2 L2A | Microsoft Planetary Computer | 12-band, 10m resolution | Global |
| EIA Form 860 | U.S. EIA | Generator inventory, plant locations, fuel type, capacity, operating year | US |
| EIA Form 923 | U.S. EIA | Generation and fuel consumption | US |
| AEO/NEMS Forecasts | EIA API v2 | Price and capacity projections | US |
| OSM Energy Polygons | OpenStreetMap | 1.66M energy infrastructure boundaries | Global |
| OlmoEarth Pretrain | Allen Institute / HuggingFace | Foundation model pretrained on Sentinel-2 | Global |
| Global Solar Atlas | World Bank / Solargis | GHI/DNI irradiance | Global |
| Global Wind Atlas | World Bank / DTU | Wind speed/power density | Global |
| NREL NSRDB | NREL | Solar resource (GHI/DNI) | US + some international |
| NREL Wind Toolkit | NREL | Wind speed/direction at hub height | US |
| USGS 3DEP / SRTM | USGS / NASA | Elevation, slope, aspect | Global |
| NLCD | USGS | Land cover classification (30m) | US |
| HydroSHEDS | WWF | River networks, watersheds | Global |
| FEMA Flood Maps | FEMA | Flood risk zones | US |
| USGS Heat Flow | USGS | Geothermal heat flow | US |

---

## 12. Infrastructure & Stack

| Layer | Tool/Technology |
|---|---|
| Foundation Model | OlmoEarth (allenai/olmoearth_pretrain), ~97M params BASE |
| ML Framework | rslearn + PyTorch Lightning |
| Training Hardware | NVIDIA GPU (local, CUDA) |
| Experiment Tracking | Weights & Biases |
| Dataset Hosting | HuggingFace Datasets |
| Backend API | FastAPI (Python) |
| Frontend | Streamlit |
| LLM Inference | vLLM (local) or NVIDIA NIM (cloud) |
| Agent Protocol | Model Context Protocol (MCP) |
| Containerization | Docker / Docker Compose (CPU + GPU) |
| Satellite Access | Microsoft Planetary Computer STAC API |
| Energy Data | EIA API v2 |

---

## 13. Limitations & Generalizability

### Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Temporal snapshot** | Embeddings from 2022-2023; land changes not reflected | Re-extract periodically |
| **1.28km resolution** | Two different sites within 1.28km get same embedding | Fine for regional screening, not parcel-level |
| **Correlation ≠ causation** | XGBoost learns where plants ARE, not necessarily where they SHOULD be | Ablation + SHAP + expert alignment validate physical signals |
| **Subsurface invisible** | Geothermal heat flow, soil composition not in imagery | Factor engine adds external data sources |
| **Bias toward existing patterns** | Training on where plants exist biases toward wealthy countries with policy incentives | Cross-region transfer test validates physics-based generalization |
| **Not a substitute for site surveys** | Screening tool, not autonomous site selector | Frame as "decision support system" |

### Generalizability Beyond Renewable Energy

The framework is domain-agnostic — the pipeline (satellite imagery → foundation model embeddings → configurable factor scoring → suitability prediction) applies to any location-dependent infrastructure:

| Application | What Embeddings Capture | Additional Factors Needed |
|------------|------------------------|--------------------------|
| EV charging stations | Urban density, road networks, commercial zones | Traffic data, grid capacity |
| Battery storage | Industrial zones, grid proximity | Electricity price differentials |
| Data centers | Flat land, infrastructure access, flood risk | Power costs, fiber connectivity, cooling |
| Agricultural facilities | Soil quality proxy, water access, terrain | Crop suitability data, market access |
| Mining sites | Geological surface indicators, terrain | Subsurface surveys, mineral maps |
| Telecommunications towers | Terrain, population density proxy | Coverage models, existing towers |

This generalizability strengthens the thesis contribution: the renewable energy case study validates the methodology, while the framework itself is a reusable tool for geospatial siting problems.

---

## 14. NEMS Valuation (Future Work)

Once site suitability scoring is complete, integrate NEMS-based economic valuation:
- NPV (Net Present Value) of projected energy generation
- IRR (Internal Rate of Return) based on AEO price forecasts
- LCOE (Levelized Cost of Energy) vs regional electricity prices
- Risk-adjusted valuation incorporating climate and grid risk scores

NEMS is open source (github.com/EIAgov/NEMS) and provides LCOE/LACE/value-cost ratio modeling across 25 US supply regions.

---

## 15. References

### Representation Learning & Transfer
- TIML: Task-Informed Meta-Learning for Agriculture (Tseng, Kerner, Rolnick, 2022) — ICML Workshop. Satellite embeddings + meta-learned classifier for cross-region agricultural classification. *Methodological precedent for our approach.*
- SatCLIP: Location encoders from satellite imagery (Klemmer et al., 2023) — Frozen satellite embeddings + linear classifiers, +8-12% over location-only baselines.
- SSL4EO: Self-supervised pretraining for Earth observation (Wang et al., 2022) — Pretrained satellite encoders boost land use classification 10-20%.
- Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021) — CLIP. Frozen embeddings + linear probes paradigm.
- CropHarvest: A global satellite dataset for crop type classification (Tseng et al., 2021) — NeurIPS Datasets and Benchmarks Track.

### Foundation Models & Earth Observation
- OlmoEarth: https://github.com/allenai/olmoearth_pretrain
- OlmoEarth Paper: arXiv 2511.13655
- OlmoEarth Platform Launch: BusinessWire, Nov 2024
- Prithvi-EO-2.0: arXiv 2412.02732
- Clay Foundation Model: Development Seed, https://developmentseed.org/projects/clay/
- Towards Responsible Geospatial Foundation Models: Nature Machine Intelligence, 2025

### Site Suitability & Detection
- Global Renewables Watch: arXiv 2503.14860, March 2025
- Deep Learning Ensemble for Solar Potential Mapping: Springer, 2025
- Multi-criteria GIS for Solar & Wind Site Selection: ScienceDirect, 2025
- Global Spatial Suitability Mapping with XAI: MDPI IJGI, 11(8), 422, 2022
- Wind-Solar-Hydrogen Storage Site Selection: ScienceDirect, 2025
- Hybrid Renewable Energy Systems in Spain: Frontiers Energy Research, 2025
- Smart GIS MCDM for Solar-Wind Hybrid: Springer, 2025
- Deep Learning for Variable Renewable Energy: ACM Computing Surveys
- NREL Technical Potential & Supply Curves: NREL/TP-6A20-91900, 2024

### Risk Analytics
- TUV SUD Emerging Renewable Energy Risks: 2025
- Climate-Related Financial Risk on Energy Investments: ScienceDirect
- IRENA Renewable Energy in Climate Change Adaptation: April 2025
- U.S. Renewables Outlook 2026: POWER Magazine
- WEF Global Energy 2026: World Economic Forum
- Climate Transition Risks and Energy Sector: NBER Working Paper 33413

### Data Sources & Platforms
- EIA Open Data: https://www.eia.gov/opendata/
- EIA AEO 2025 LCOE Report: https://www.eia.gov/outlooks/aeo/electricity_generation/
- NEMS Open Source: https://github.com/EIAgov/NEMS
- Microsoft Planetary Computer: https://planetarycomputer.microsoft.com/
- Global Solar Atlas: https://globalsolaratlas.info/
- Global Wind Atlas: https://globalwindatlas.info/
- IRENA Global Atlas: https://globalatlas.irena.org/
- ESMAP RE Resource Mapping: https://www.esmap.org/re_mapping
- Gridded Climate Datasets for Wind/Solar Yield: Nature, 2025

---

## Appendix A: Mentor Communication

Dear Professor Ganguly,

I hope this message finds you well. I wanted to provide a detailed update on my senior thesis research project, OpenEnergy-Engine, and share the current state of the platform, datasets, and infrastructure.

### Project Overview

**Repository:** github.com/2imi9/OpenEnergy-Engine
**Goal:** AI-powered renewable energy site suitability scoring and risk analytics using geospatial foundation models

### Research Direction

After extensive experimentation with pixel-level energy infrastructure detection (29 runs, best F1=0.46), I pivoted the project direction based on a key insight: OlmoEarth at 10m Sentinel-2 resolution excels at landscape-scale land characterization, not sub-pixel object detection. The literature review confirmed no published work combines geospatial foundation model embeddings with configurable multi-criteria suitability scoring — this is the research gap we address.

**Current approach:** Train on historical plant locations globally (OSM + EIA 860) as positive examples of site suitability. Use OlmoEarth embeddings as learned landscape features alongside external data (solar irradiance, wind speed, terrain, grid proximity). Validate with retroactive prediction (train on pre-2020 plants, test on 2021-2025) and cross-region transfer (train US/EU, test India/Brazil/Africa).

### Platform Status

The application platform (web API, MCP server, LLM integration, Streamlit dashboard, NEMS valuation engine) is fully built and waiting for the suitability model. The factor engine and scoring pipeline are the current development focus.

### Datasets

Sentinel-2 (global, 10m), EIA 860/923 (US plants), OSM (1.66M global energy polygons), NREL NSRDB/Wind Toolkit (solar/wind resource), USGS elevation, and others detailed in the thesis document.

### Questions for Discussion

1. Should we prioritize solar+wind first, or attempt all energy types simultaneously?
2. Is retroactive prediction (pre-2020 train, 2021+ test) sufficient validation for the thesis committee?
3. Do you have recommendations for cross-region transfer test locations?
4. Any preferred scope for the NEMS valuation integration?

I welcome any feedback on the research direction. I believe combining foundation model embeddings with configurable factor scoring addresses a genuine gap in the literature and creates a stronger contribution than competing with existing high-resolution detection systems.

Best regards,
Ziming (Frank) Qi
Northeastern University, Millennium Fellowship Research

---

## Appendix B: Detection Phase (Lessons Learned)

### Summary of 29 Detection Experiments

The detection phase (pixel-level segmentation of energy infrastructure) produced valuable technical insights that informed the project pivot:

1. **Dataset imbalance is the #1 killer:** 94% background-only windows caused 18 consecutive failed runs (model learned to predict all-background)
2. **Label quality matters more than model architecture:** EIA centroid circles (500m buffer) provide zero spectral contrast at 10m. OSM polygon labels show clear signal (B08 ratio 0.72 vs 0.97)
3. **Frozen encoder is essential for small datasets:** Unfreezing OlmoEarth's pretrained encoder with <1000 training windows always caused collapse. Feature extraction (frozen encoder) achieved F1=0.46.
4. **10m resolution ceiling:** Solar panels are ~2m, wind turbines are ~5m. At 10m pixels, these are sub-pixel features. Detection accuracy is fundamentally limited.
5. **OlmoEarth learns useful landscape features:** The frozen encoder's success proves the 768-dim embeddings capture meaningful land characterization, even though the decoder task (pixel segmentation) was too hard.

### Key Metrics from Detection Phase

| Metric | Best Value | Run | Notes |
|---|---|---|---|
| F1 (micro) | 0.46 | 22 | Frozen encoder + OSM labels |
| Precision | ~42% | 22 | Many false positives |
| Recall | ~53% | 21 | Finds about half of real pixels |
| val_loss | 0.826 | 21 | Cross-entropy + dice |

### Archived Artifacts

- 18 model YAML configs in `archive/detection_configs/`
- Detection scripts in `archive/detection_scripts/`
- Best checkpoint: `project_data/openenergyengine/run_21_frozen/best.ckpt`
- Dataset: 9,292 windows with Sentinel-2 imagery + OSM labels in `dataset/windows/ready/`
