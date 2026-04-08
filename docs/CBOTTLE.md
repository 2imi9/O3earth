# cBottle Integration

**Status:** In progress — `feature/cbottle-integration` branch

## What

Replace NASA POWER (~50km) with NVIDIA's [cBottle](https://github.com/NVlabs/cBottle) (~7km) for higher-resolution climate inputs to the factor engine.

## Why

NASA POWER smooths out local terrain effects on wind and solar at 50km resolution. cBottle's km-scale downscaling captures valley winds, coastal effects, and orographic precipitation that our current pipeline misses. Biggest impact expected for **wind** siting where local terrain matters most.

## What's Done

- `src/data_clients/cbottle.py` — Drop-in client that outputs the same keys as NASA POWER (`ghi_kwh_m2_day`, `wind_speed_ms`, `avg_temp_c`, `precipitation_mm_day`). Uses Earth2Studio API when available, falls back to native cBottle.

## What's Left

- [ ] Wire into `realtime.py` as optional high-res source (fallback to NASA POWER when no GPU)
- [ ] Add `/api/suitability?resolution=high` parameter to trigger cBottle
- [ ] Test unit conversions (rsds W/m2 → GHI kWh/m2/day, pr kg/m2/s → mm/day)
- [ ] Benchmark: compare factor engine scores with NASA POWER vs cBottle inputs
- [ ] Add to platform UI as toggle

## Requirements

- NVIDIA GPU (A100/H100 recommended, ~24GB VRAM minimum)
- `pip install earth2studio` or clone cBottle from source
- `pip install healpy` for HEALPix grid lookups

## Limitations

- cBottle is **historical downscaling only** (trained 1970-2022), not SSP projections
- For future climate risk, still need CMIP6 deltas on top of cBottle baselines
- GPU requirement means this is an optional upgrade, not default — aligns with O3 EartH's "no GPU needed" philosophy

## Contact

Noah Brenowitz (NVlabs) confirmed cBottle is effectively open source and encouraged development use. Conversation ongoing via LinkedIn.
