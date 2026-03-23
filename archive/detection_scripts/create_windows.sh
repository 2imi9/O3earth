#!/usr/bin/env bash
# create_windows.sh — Add rslearn windows at US energy infrastructure hotspots
#
# QUICK START: This covers 21 hand-picked regions (~300-500 windows).
# For FULL COVERAGE of all ~2000+ EIA plants, use instead:
#   python scripts/create_windows_from_eia.py --csv source_data/eia860/3_1_Generator_Y2023.csv
#
# Each region is chosen to cover clusters of EIA 860 power plants.
# Uses UTM projection, 10m resolution, 128px grid, summer 2023 imagery.

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset}"
START="2023-05-01T00:00:00+00:00"
END="2023-09-01T00:00:00+00:00"
GRID=128
RES=10

add_region() {
    local name="$1" box="$2"
    echo ">>> Adding windows: ${name} (${box})"
    rslearn dataset add_windows \
        --root "$DATASET_ROOT" \
        --group default \
        --utm --resolution "$RES" \
        --src_crs EPSG:4326 \
        --box="$box" \
        --start "$START" --end "$END" \
        --grid_size "$GRID"
}

# ── Solar hotspots ────────────────────────────────────────────────────────────
add_region "CA-Mojave-Solar"         "-117.5,34.5,-115.5,35.5"
add_region "CA-Imperial-Solar"       "-116.0,32.5,-115.0,33.5"
add_region "CA-SanJoaquin-Solar"     "-120.5,35.0,-119.5,36.0"
add_region "AZ-Phoenix-Solar"        "-113.0,32.5,-112.0,33.5"
add_region "NV-Eldorado-Solar"       "-115.5,35.5,-114.5,36.5"
add_region "TX-WestTexas-Solar"      "-102.5,31.0,-101.0,32.5"
add_region "NC-Solar-Belt"           "-79.5,35.0,-78.0,36.0"

# ── Wind hotspots ─────────────────────────────────────────────────────────────
add_region "TX-Panhandle-Wind"       "-101.5,34.0,-100.0,35.5"
add_region "TX-SouthWind"            "-98.0,27.0,-97.0,28.0"
add_region "IA-Wind-Corridor"        "-94.5,41.5,-93.0,42.5"
add_region "KS-Wind-Belt"            "-99.0,37.5,-97.5,38.5"
add_region "OK-Wind"                 "-98.5,35.5,-97.0,36.5"
add_region "MN-Wind"                 "-96.0,43.5,-94.5,44.5"

# ── Coal / Gas / Nuclear / Hydro ──────────────────────────────────────────────
add_region "WV-Coal-Country"         "-81.5,38.0,-80.5,39.0"
add_region "PA-Coal-Gas"             "-80.0,40.0,-79.0,41.0"
add_region "IL-Nuclear-Coal"         "-89.5,41.0,-88.0,42.0"
add_region "TN-TVA-Hydro"           "-85.0,35.0,-84.0,36.0"
add_region "WA-Columbia-Hydro"       "-120.5,46.5,-119.5,47.5"
add_region "LA-GasRefinery"          "-93.5,30.0,-92.5,31.0"

# ── Mixed / Geothermal / Storage ──────────────────────────────────────────────
add_region "CA-Geysers-Geothermal"   "-123.0,38.5,-122.5,39.0"
add_region "NJ-Mixed-Coast"          "-74.5,39.5,-74.0,40.0"

echo ""
echo "Done. Windows created in ${DATASET_ROOT}/windows/default/"
echo "Next step: bash scripts/build_dataset.sh"
