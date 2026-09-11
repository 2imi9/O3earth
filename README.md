# O3 EartH

**Geospatial Site Suitability Assessment Using Foundation Model Embeddings**

O3 EartH scores renewable energy site suitability from frozen [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) satellite embeddings with lightweight classifiers. Scoring runs on CPU.

> Ziming Qi | Northeastern University

## Overview

**Question.** Do frozen embeddings from a geospatial foundation model carry information about renewable energy site suitability that simple geographic features do not already provide, and does that information survive when the model is tested on countries it never saw during training?

**Method.** 8,000 locations across 212 countries and 4 energy types (solar, wind, hydro, geothermal). Each location is a Sentinel-2 L2A patch (12 bands, 128x128 px at 10 m). Solar, wind and hydro use 4 seasonal scenes; geothermal uses a single scene. Patches pass through a frozen OlmoEarth BASE encoder to give one 768-dim vector per location, and XGBoost classifies on top. Positives are existing energy sites; negatives are random global locations matched by energy type count. The encoder is never fine-tuned, so scoring a stored embedding is a CPU operation.

**Answer.** Under leave-one-country-out spatial cross-validation across 63 countries, AUC is **0.867 ± 0.114**. I report this as the headline number because it is the evaluation that removes geographic leakage. Standard 5-fold cross-validation on the same embeddings gives **0.911 ± 0.015**. The gap of 4.4 points between the two is the leakage that random splits hide.

**What the embeddings add is real but modest.** Features derived only from latitude and longitude already reach **0.852**. Embeddings alone reach **0.913**, and embeddings plus those geographic features reach **0.927**. So the foundation model contributes about **+6.1 points** over geography, not the large jump a geography-only baseline near chance would imply. An earlier version of this README reported a 0.579 geographic baseline; that number is not supported by any result file in this repository and has been removed.

**What is not identifiable.** The labels separate existing energy sites from random locations, so the classifier learns "does this landscape resemble places where plants have been built". That is not the same as economic viability, permitting feasibility, or output forecasting, and this dataset cannot distinguish those. Geothermal rests on 260 samples, too few to treat its per-type score as stable. The claim that multi-temporal (T=4) embeddings beat single-scene (T=1) embeddings is **not established here**: [scripts/test_multitemporal.py](scripts/test_multitemporal.py) implements the comparison but writes no saved output, so the repository contains no T=1 result.

**Two runs, two numbers.** A later re-run stored in [suitability_results_v3.json](results/suitability/suitability_results_v3.json) reports a higher spatial CV AUC of 0.904 and overall CV of 0.924. No script in this repository reproduces that file, so the documented 0.867 / 0.911 run is the one to rely on. Both are listed below rather than the better one alone.

## Key results

| What | Number | Note |
|------|--------|------|
| Spatial CV, leave-one-country-out, 63 countries | **0.867 ± 0.114** | Conservative headline. No leakage from nearby train points. [VALIDATION.md](docs/VALIDATION.md) |
| 5-fold stratified CV, embeddings only | **0.911 ± 0.015** | Folds 0.911 / 0.898 / 0.894 / 0.917 / 0.934. [suitability_results.json](results/suitability/suitability_results.json) |
| Ablation: lat/lon-derived features only | 0.852 | 8 features synthesized from coordinates, not measured resource data. [train_suitability.py:194](scripts/train_suitability.py) |
| Ablation: OlmoEarth embeddings only | 0.913 | +6.1 points over the coordinate baseline. [suitability_results.json](results/suitability/suitability_results.json) |
| Ablation: embeddings + coordinate features | 0.927 | Best configuration. [suitability_results.json](results/suitability/suitability_results.json) |
| Random-label control | 0.497 | Expected ~0.50. Model does not memorize noise. [VALIDATION.md](docs/VALIDATION.md) |
| Regional spread, 6 continents | 0.866 (Asia) to 0.943 (South America) | Per-continent AUC. [suitability_results.json](results/suitability/suitability_results.json) |
| Re-run: spatial CV, 63 countries | 0.904 | Higher than the documented run. Not reproducible from code in this repo. [suitability_results_v3.json](results/suitability/suitability_results_v3.json) |
| Re-run: overall CV | 0.924 | Same caveat as above. [suitability_results_v3.json](results/suitability/suitability_results_v3.json) |
| Re-run, per type: solar / geothermal / hydro / wind | 0.959 / 0.930 / 0.918 / 0.898 | n = 3,205 / 260 / 1,281 / 3,254. Random-split CV, not spatial. [suitability_results_v3.json](results/suitability/suitability_results_v3.json) |

Re-run the ablation and 5-fold cross-validation on the embeddings committed here:

```bash
python scripts/train_suitability.py \
  --embeddings data/embeddings_v3/embeddings.npy \
  --metadata data/embeddings_v3/embeddings_meta.csv \
  --cv-folds 5 \
  --output-dir results/rerun
```

This runs the same pipeline, but not on the same inputs. The committed [suitability_results.json](results/suitability/suitability_results.json) was produced from the earlier `embeddings/` set, which lives on Hugging Face and is excluded from this repository by `.gitignore`. Download that set to reproduce 0.911 and the ablation numbers exactly. `--output-dir` points at a new directory so the committed result file is not overwritten.

There is **no script in this repository that reproduces the leave-one-country-out number or the v3 per-type numbers**. Both were produced outside the committed code.

## Status

- **Done.** Embedding extraction pipeline, 8,000-location dataset over 212 countries, trained XGBoost classifiers per energy type, ablation and 5-fold CV with saved result files, and a FastAPI + Streamlit platform with MCP tools that scores stored embeddings on CPU.
- **Open.** No committed script regenerates the spatial CV or the v3 results file, and the v1 `embeddings/` set behind the documented ablation and 5-fold numbers is gitignored, so those numbers cannot be re-derived from a fresh clone alone. The T=1 vs T=4 multi-temporal comparison needs a run that saves its output. Dataset is versioned by directory convention only: the Hugging Face repo holds `embeddings/` + `models/` alongside `embeddings_v3/` + `models_v3/`, with no tags or release versions in either repository.
- **Known limitations.** Negative samples are random locations, so scores measure landscape resemblance to built sites, not viability. Geothermal has 260 samples. Spatial CV standard deviation is wide (±0.114), so per-country performance varies substantially. [docs/VALIDATION.md](docs/VALIDATION.md), [CITATION.cff](CITATION.cff) and the Hugging Face card still quote the older run's numbers and are not yet reconciled with the v3 file.

## Platform

| Page | What it does |
|------|-------------|
| **AI Chat** | NVIDIA NIM LLM with system knowledge |
| **Site Selection** | Map, pick location, Factor Engine + ML scores |
| **Climate Risk** | NASA POWER data + IPCC AR6 SSP projections |

The Factor Engine scores 19 configurable factors from live APIs, separate from the ML path. MCP tools are available for programmatic access. Details in [PLATFORM.md](docs/PLATFORM.md).

## Data sources

| Source | Data | Auth |
|--------|------|------|
| NASA POWER | Solar GHI, wind speed, temperature, cloud, precipitation | None |
| Open-Elevation | Terrain slope and gradient | None |
| Open-Meteo Flood | River discharge | None |
| USGS Earthquake | Seismic activity | None |
| EIA API v2 | US power plant data | API key |
| Planetary Computer | Sentinel-2 imagery | None |

## Install

```bash
git clone https://github.com/2imi9/O3earth.git
cd O3earth
pip install -r requirements.txt
```

Optional API keys:

```bash
cp platform/.env.example platform/.env
```

| Variable | Required for | Get one |
|----------|-------------|---------|
| `EIA_API_KEY` | US power plant data | [eia.gov/opendata](https://www.eia.gov/opendata/register.php) |
| `NVIDIA_API_KEY` | AI Chat | [build.nvidia.com](https://build.nvidia.com/) |

## Quick start

Docker:

```bash
cd platform
docker compose up --build
```

Manual:

```bash
cd platform
uvicorn api.main:app --port 8000        # terminal 1
streamlit run ui/app.py --server.port 8501   # terminal 2
```

Open [localhost:8501](http://localhost:8501). Site Selection and Climate Risk work without API keys. AI Chat requires `NVIDIA_API_KEY`.

## Repository layout

```
src/factors/        19 scoring factors (rule-based engine)
src/scoring/        Suitability engine
src/data_clients/   API clients
src/mcp/            MCP tools + handlers
src/llm/            NVIDIA NIM client
platform/api/       FastAPI backend
platform/ui/        Streamlit frontend
scripts/            Data pipeline, embedding extraction, training
data/embeddings_v3/ 8,000 x 768 embeddings + metadata
results/            Trained models and metrics
docs/               Validation and platform docs
```

## Docs

- [docs/VALIDATION.md](docs/VALIDATION.md): ablation, cross-validation, spatial CV, temporal validation, sanity checks
- [docs/RESULTS.md](docs/RESULTS.md): result summary
- [docs/PLATFORM.md](docs/PLATFORM.md): architecture and MCP tools

Dataset and trained models on Hugging Face: [2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth). Models are inside the dataset repository under `models/` and `models_v3/`; there is no separate model page.

## References

- [OlmoEarth](https://github.com/allenai/olmoearth_pretrain): Allen Institute geospatial foundation model
- [TIML](https://arxiv.org/abs/2209.06277) (Tseng et al.): methodological precedent for embedding + classifier approach
- [SatCLIP](https://arxiv.org/abs/2311.17179) (Klemmer et al.): location embeddings from satellite imagery

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT
