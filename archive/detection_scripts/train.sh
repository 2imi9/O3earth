#!/usr/bin/env bash
# train.sh — Launch OlmoEarth fine-tuning with rslearn
#
# Prerequisite: dataset must be built and split (see build_dataset.sh, split_train_val.py)

set -euo pipefail

export PROJECT_NAME="${PROJECT_NAME:-openenergyengine}"
export RUN_NAME="${RUN_NAME:-run_00}"
export MANAGEMENT_DIR="${MANAGEMENT_DIR:-./project_data}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== OlmoEarth Fine-Tune ==="
echo "  Project:  ${PROJECT_NAME}"
echo "  Run:      ${RUN_NAME}"
echo "  Ckpt dir: ${MANAGEMENT_DIR}"
echo "  W&B mode: ${WANDB_MODE}"
echo ""

rslearn model fit --config configs/model.yaml

echo ""
echo "Training complete. Best checkpoint saved in ${MANAGEMENT_DIR}/"
echo "Next step: bash scripts/predict.sh"
