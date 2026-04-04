# O3 EartH

**Geospatial Site Suitability Assessment Using Foundation Model Embeddings**

O3 EartH uses frozen [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) embeddings with lightweight classifiers to score infrastructure site suitability — no GPU needed at inference.

> Ziming Qi | Northeastern University

## How It Works

```
Sentinel-2 (12 bands, multi-temporal) → OlmoEarth (frozen, 97M params) → 768-dim embedding → XGBoost → Score
```

- **Multi-temporal**: 4 seasonal scenes for solar/wind/hydro, single scene for geothermal
- **Extract once**: OlmoEarth converts satellite patches into 768-dim landscape fingerprints
- **Score instantly**: XGBoost classifies on CPU in milliseconds

## Key Results

| Method | AUC |
|--------|-----|
| Geography only (lat/lon) | 0.579 |
| OlmoEarth T=1 (single scene) | 0.907 |
| **OlmoEarth T=multi (seasonal)** | **0.924** |
| **Spatial CV (leave-one-country-out, 63 countries)** | **0.904** |

| Energy Type | AUC |
|-------------|-----|
| Solar | 0.959 |
| Geothermal | 0.930 |
| Hydro | 0.918 |
| Wind | 0.898 |

8,000 samples across 212 countries, 4 energy types. Full validation in **[VALIDATION.md](docs/VALIDATION.md)**.

## Platform

Web application with three components:

| Page | What it does |
|------|-------------|
| **AI Chat** | LLM with full system knowledge (NVIDIA NIM or local Gemma 4) |
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

### 1. Clone

```bash
git clone https://github.com/2imi9/O3earth.git
cd O3earth
```

### 2. Configure API keys (optional)

```bash
cp platform/.env.example platform/.env
# Edit with your keys (optional — Site Selection works without them)
```

| Variable | Required For | Get One |
|----------|-------------|---------|
| `EIA_API_KEY` | US power plant data | [eia.gov/opendata](https://www.eia.gov/opendata/register.php) |
| `NVIDIA_API_KEY` | AI Chat (cloud) | [build.nvidia.com](https://build.nvidia.com/) |

### 3. AI Chat backend (optional)

AI Chat supports two backends — pick one or skip (Site Selection and Climate Risk work without it):

**Option A: NVIDIA NIM (cloud, no GPU needed)**

Set `NVIDIA_API_KEY` in `platform/.env`. Uses `openai/gpt-oss-20b` via NVIDIA's hosted API.

**Option B: Local vLLM ([Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it))**

```bash
pip install vllm
```

Uncomment the vLLM settings in `platform/.env`:

```bash
VLLM_MODEL=google/gemma-4-E2B-it
VLLM_DTYPE=float16
VLLM_GPU_MEMORY=0.9
```

Gemma 4 E2B is 2.3B effective params — runs on a laptop with ~4GB VRAM.

The AI Chat page lets you switch between backends in-app when both are available.

### 4. Run

**Docker (recommended):**

```bash
cd platform
docker compose up --build
```

**With GPU + local vLLM:**

```bash
cd platform
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**Manual (no Docker):**

```bash
pip install -r requirements.txt

# Terminal 1 — API
cd platform
uvicorn api.main:app --port 8000

# Terminal 2 — UI
cd platform
streamlit run ui/app.py --server.port 8501
```

### 5. Open

Go to [localhost:8501](http://localhost:8501). Site Selection and Climate Risk work immediately.

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
