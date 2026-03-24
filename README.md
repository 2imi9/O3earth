# O3 EartH

**Geospatial Site Suitability Assessment Using Foundation Model Embeddings**

O3 EartH uses frozen [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) embeddings with lightweight classifiers to score infrastructure site suitability — no GPU needed at inference.

> Ziming Qi | Northeastern University

## How It Works

```
Sentinel-2 (12 bands) → OlmoEarth (frozen, 97M params) → 768-dim embedding → XGBoost → Score
```

- **Extract once**: OlmoEarth converts satellite patches into 768-dim landscape fingerprints
- **Score instantly**: XGBoost classifies on CPU in milliseconds
- **Compare methods**: Factor Engine (real-time APIs) vs ML Model (satellite embeddings)

## Key Result

| Method | AUC |
|--------|-----|
| Geography only (lat/lon) | 0.579 |
| OlmoEarth embeddings | 0.902 |
| **Spatial CV (leave-one-country-out)** | **0.867** |

8,000 samples across 212 countries, 4 energy types. Full validation methodology in **[VALIDATION.md](docs/VALIDATION.md)** — includes spatial CV, temporal validation, overfitting diagnosis, and reproducibility instructions.

## Platform

Web application with three components:

| Page | What it does |
|------|-------------|
| **AI Chat** | NVIDIA NIM LLM with full system knowledge |
| **Site Selection** | Map → pick location → Factor Engine + ML scores |
| **Climate Risk** | NASA POWER data + IPCC AR6 SSP projections |

MCP tools available for programmatic access. Details in [PLATFORM.md](docs/PLATFORM.md).

## Data Sources

| Source | Data | Auth |
|--------|------|------|
| NASA POWER | Solar GHI, wind speed, temperature, cloud, precipitation | None |
| Open-Elevation | Terrain slope and gradient | None |
| Open-Meteo Flood | River discharge | None |
| USGS Earthquake | Seismic activity | None |
| EIA API v2 | US power plant data | API key |
| Planetary Computer | Sentinel-2 imagery | None |

## Quick Start

```bash
# Clone and install
git clone https://github.com/2imi9/O3earth.git
cd O3earth
pip install -r requirements.txt

# Run the platform (no GPU needed)
cd platform
uvicorn api.main:app --port 8000 &
streamlit run ui/app.py --server.port 8501
```

Dataset and pre-trained models on HuggingFace: [2imi9/O3earth](https://huggingface.co/datasets/2imi9/O3earth)

## Project Structure

```
O3earth/
├── src/
│   ├── factors/           # 19 configurable scoring factors
│   ├── scoring/           # Suitability engine
│   ├── data_clients/      # Real-time API clients
│   ├── mcp/               # MCP tools + handlers
│   └── llm/               # NVIDIA NIM + vLLM
├── platform/
│   ├── api/               # FastAPI backend
│   └── ui/                # Streamlit frontend
├── scripts/               # Data pipeline + training
├── data/                  # Datasets + embeddings
└── results/               # Trained models + metrics
```

## References

- [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) — Allen Institute geospatial foundation model
- [TIML](https://arxiv.org/abs/2209.06277) (Tseng et al.) — methodological precedent for embedding + classifier approach
- [SatCLIP](https://arxiv.org/abs/2311.17179) (Klemmer et al.) — location embeddings from satellite imagery

## License

MIT
