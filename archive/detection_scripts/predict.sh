#!/usr/bin/env bash
# predict.sh — Run OlmoEarth inference on a new region
#
# Usage:
#   bash scripts/predict.sh                                    # default: Mojave test area
#   bash scripts/predict.sh --box="-116.5,34.5,-115.5,35.5"   # custom bbox
#   bash scripts/predict.sh --name my_region --box="..."       # custom name + bbox

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset}"
MANAGEMENT_DIR="${MANAGEMENT_DIR:-./project_data}"

# Defaults
BOX="-116.5,34.5,-115.5,35.5"
NAME="predict_region"
START="2023-05-01T00:00:00+00:00"
END="2023-09-01T00:00:00+00:00"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --box=*) BOX="${1#*=}" ;;
        --name=*) NAME="${1#*=}" ;;
        --start=*) START="${1#*=}" ;;
        --end=*) END="${1#*=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "=== Predict: ${NAME} ==="
echo "  Box: ${BOX}"
echo ""

# Add prediction windows
echo ">>> Adding prediction windows..."
rslearn dataset add_windows \
    --root "$DATASET_ROOT" \
    --group predict \
    --utm --resolution 10 \
    --src_crs EPSG:4326 \
    --box="$BOX" \
    --start "$START" --end "$END" \
    --grid_size 128

# Prepare + ingest for prediction windows
echo ">>> Preparing & ingesting..."
rslearn dataset prepare --root "$DATASET_ROOT" --workers 16
rslearn dataset ingest --root "$DATASET_ROOT" --workers 16
rslearn dataset materialize --root "$DATASET_ROOT" --workers 16 --ignore-errors

# Run prediction
echo ">>> Running model predict..."
export PROJECT_NAME="${PROJECT_NAME:-openenergyengine}"
export RUN_NAME="${RUN_NAME:-run_00}"
export MANAGEMENT_DIR="$MANAGEMENT_DIR"

rslearn model predict --config configs/model.yaml

echo ""
echo "Predictions written to ${DATASET_ROOT}/windows/predict/*/layers/output/"
