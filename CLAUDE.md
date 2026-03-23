# OpenEnergy-Engine — Site Suitability & Risk Analytics

## Project Identity
**Repo:** `2imi9/OpenEnergy-Engine` (this is the model/engine repo)
**Platform repo:** `2imi9/OpenEnergy-Engine` (has web API, MCP, LLM, dashboard — waiting for this model)
**Researcher:** Ziming (Frank) Qi, Northeastern University, Millennium Fellowship
**Mentor:** Professor Auroop Ratan Ganguly

## What This Project Does

AI-powered renewable energy **site recommendation system** with configurable risk analytics:

1. **Land Characterization** — OlmoEarth ViT extracts landscape features from Sentinel-2 imagery
2. **Factor Engine** — Configurable scoring factors per energy type (solar, wind, hydro, geothermal, gas)
3. **Site Scoring** — Multi-factor suitability score for any location
4. **Risk Analytics** — Climate, grid, environmental, economic risk assessment
5. **NEMS Valuation** — (later) NPV/IRR/LCOE using EIA AEO projections

## OlmoEarth Model

**Input:** 12-band Sentinel-2 image patch (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09)
- Shape: `[batch, 12, H, W]` where H,W are multiples of patch_size (4)
- Normalized with per-band `(val - (mean - 2*std)) / (4*std)`

**Output (as feature extractor):** Multi-scale feature maps from ViT encoder
- Shape: `[batch, 768, H/4, W/4]` for BASE model
- These are landscape-level features (land cover type, vegetation density, terrain texture, built-up area, water bodies)

**Model sizes:**
- `OLMOEARTH_V1_NANO` — fastest, debugging
- `OLMOEARTH_V1_TINY` — lightweight
- `OLMOEARTH_V1_BASE` — 97M params, recommended (what we use)
- `OLMOEARTH_V1_LARGE` — best quality, needs A100+

**Our usage:** Feature extractor for land characterization. The 768-dim features feed into the factor engine as one input alongside external data (solar irradiance, wind speed, terrain, etc.).

## Factor Engine Architecture

Each energy type has toggleable factors:

### General Factors (always on)
- Land cover / land use classification
- Terrain (slope, aspect, elevation)
- Grid proximity (distance to transmission lines)
- Road access
- Environmental constraints (protected areas, wetlands)
- Flood risk

### Solar-Specific
- GHI / DNI (solar irradiance)
- Cloud coverage frequency
- Precipitation patterns
- Temperature effects on panel efficiency

### Wind-Specific
- Wind velocity and direction
- Turbulence intensity
- Terrain roughness
- Wake effects

### Hydro-Specific
- Water source stability
- Water flow speed
- Elevation drop / head
- Watershed health

### Geothermal-Specific
- Heat flow density
- Lithology / rock type
- Fault proximity
- Seismic activity

## Directory Structure

```
OpenEnergy-Engine/
├── CLAUDE.md                    ← THIS FILE
├── src/
│   ├── factors/                 ← Factor definitions per energy type
│   ├── scoring/                 ← Scoring engine
│   └── data_clients/            ← External data API clients
├── configs/                     ← Scoring configs
├── scripts/                     ← Utility scripts
├── data_sources/                ← Legacy EIA data source
├── archive/                     ← Old detection experiments (thesis reference)
├── dataset/                     ← rslearn dataset (Sentinel-2 imagery)
├── project_data/                ← Model checkpoints
└── data/                        ← OSM polygons, source data
```

## Key Data Sources & APIs

| Source | What | Access |
|--------|------|--------|
| EIA API v2 | Plant locations, generation, prices, forecasts | `EIA_API_KEY` |
| Sentinel-2 L2A | 12-band satellite imagery, 10m | Planetary Computer |
| NREL NSRDB | Solar irradiance (GHI/DNI) | API key needed |
| NREL Wind Toolkit | Wind speed/direction | API key needed |
| USGS 3DEP / SRTM | Elevation, slope, aspect | Free |
| OSM | Energy infrastructure polygons | Downloaded (1.66M) |
| NLCD | Land cover classification | Free |

## Best Detection Checkpoint
- Path: `project_data/openenergyengine/run_21_frozen/best.ckpt`
- OlmoEarth BASE, frozen encoder, F1=0.46
- Use as feature extractor, not pixel detector

## Priority Order
1. Site recommendation system (factor engine + scoring)
2. Risk analytics
3. NEMS valuation (last)
