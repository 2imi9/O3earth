[![O3 EartH Demo](https://img.youtube.com/vi/vaiKRWlHqPc/maxresdefault.jpg)](https://www.youtube.com/watch?v=vaiKRWlHqPc)

# Platform Architecture

## Overview

O3 EartH is a web application (FastAPI + Streamlit) for renewable energy site suitability assessment. No GPU required at runtime.

## Pages

| Page | Description |
|------|-------------|
| **AI Chat** | NVIDIA NIM-powered assistant with system knowledge of the full architecture |
| **Site Selection** | Interactive map with dual scoring: Factor Engine (real-time APIs) vs OlmoEarth ML (satellite embeddings). Compare locations side by side. |
| **Climate Risk** | NASA POWER current data + IPCC AR6 SSP scenario projections |

## Scoring Methods

**Factor Engine** — Rule-based scoring from real-time API data:

| API | Data | Energy Types |
|-----|------|-------------|
| NASA POWER | GHI, wind speed, temperature, cloud cover, precipitation | Solar, Wind |
| Open-Elevation | Terrain slope and gradient | Solar, Wind, Hydro |
| Open-Meteo Flood | River discharge | Hydro |
| USGS Earthquake | Seismic activity | Geothermal |
| EIA API v2 | US power plant locations and capacity | All |

**OlmoEarth ML** — Satellite embedding similarity to known energy sites:

```
Sentinel-2 patch → OlmoEarth (frozen, 768-dim) → XGBoost classifier → suitability probability
```

Pre-computed embeddings for 8,000 global locations. AUC 0.867 (spatial CV).

## MCP Tools

| Tool | Description |
|------|-------------|
| `score_suitability` | Score a location for solar or wind suitability |
| `assess_climate_risk` | Climate risk assessment with SSP projections |
| `query_eia` | Query US energy plant database |
| `analyze` | LLM-powered site analysis |
| `generate_report` | Generate formatted suitability report |

## Quick Start

```bash
# Install
cd platform && pip install -r requirements.txt

# Backend
uvicorn api.main:app --port 8000

# Frontend (separate terminal)
streamlit run ui/app.py --server.port 8501
```

Environment variables in `platform/.env`:
```
NVIDIA_API_KEY=your_key
EIA_API_KEY=your_key        # optional
```
