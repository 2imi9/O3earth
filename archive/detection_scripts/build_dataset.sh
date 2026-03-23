#!/usr/bin/env bash
# build_dataset.sh — Run the rslearn prepare → ingest → materialize pipeline
#
# Prerequisite: run create_windows.sh first.
# This downloads Sentinel-2 imagery + rasterizes EIA labels for every window.

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset}"
WORKERS="${WORKERS:-16}"

echo "=== Phase 1/3: Prepare (query metadata) ==="
rslearn dataset prepare \
    --root "$DATASET_ROOT" \
    --workers "$WORKERS"

echo ""
echo "=== Phase 2/3: Ingest (download raw data) ==="
rslearn dataset ingest \
    --root "$DATASET_ROOT" \
    --workers "$WORKERS"

echo ""
echo "=== Phase 3/3: Materialize (crop & reproject into tiles) ==="
rslearn dataset materialize \
    --root "$DATASET_ROOT" \
    --workers "$WORKERS" \
    --ignore-errors

echo ""
echo "Done. Dataset materialized at ${DATASET_ROOT}/windows/"
echo "Next step: python scripts/split_train_val.py"
