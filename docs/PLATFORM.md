[![O3 EartH Demo](https://img.youtube.com/vi/vaiKRWlHqPc/maxresdefault.jpg)](https://www.youtube.com/watch?v=vaiKRWlHqPc)

# Platform Architecture

## Overview
O3 EartH runs as a web application with a FastAPI backend and Streamlit frontend. No GPU required.

## Pages

### AI Chat
- Powered by NVIDIA NIM (cloud LLM)
- System prompt includes full architecture knowledge
- Contextual: aware of user's recent suitability scores and climate risk results

### Site Selection
- Interactive map for location picking
- Dual scoring: Factor Engine (real-time APIs) vs OlmoEarth ML (satellite embeddings)
- Preset locations with pre-computed embeddings for demo
- "Compare with preset locations" ranks sites in a table

### Climate Risk
- Current climate data from NASA POWER API
- IPCC AR6 SSP scenario projections (SSP126 through SSP585)
- Extreme event probability estimates
- Resource projections (GHI, wind speed) under warming scenarios

## MCP Tools
Model Context Protocol tools for programmatic access:

| Tool | Description |
|------|-------------|
| `score_suitability` | Score lat/lon for solar or wind suitability |
| `assess_climate_risk` | Climate risk with SSP projections |
| `query_eia` | US energy plant database queries |
| `analyze` | LLM-powered analysis |
| `generate_report` | Formatted report generation |

## Real-Time Data Sources

| API | Data | Used By |
|-----|------|---------|
| NASA POWER | GHI, wind, temp, cloud, precip | Factor Engine + Climate Risk |
| Open-Elevation | Terrain slope/gradient | Factor Engine |
| Open-Meteo Flood | River discharge | Factor Engine (hydro) |
| USGS Earthquake | Seismic activity | Factor Engine (geothermal) |
| EIA API v2 | US power plants | EIA queries |

## Running

```bash
# Backend
cd platform
uvicorn api.main:app --port 8000

# Frontend (separate terminal)
streamlit run ui/app.py --server.port 8501
```

Required environment variables in `platform/.env`:
```
NVIDIA_API_KEY=your_key_here
EIA_API_KEY=your_key_here  # optional, for US energy data
```
