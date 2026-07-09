#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================="
echo "   STARTING TERRAMIND MASK2FORMER PIPELINE             "
echo "======================================================="

# 1. Add the current directory to PYTHONPATH so module imports work flawlessly
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Define our input and output directories
CONFIG_PATH="./configs/config.yaml"
SAVE_DIR="/kaggle/working/checkpoints/run_01"

echo ">>> [1/2] RUNNING TRAINING..."
python scripts/train.py \
    --config $CONFIG_PATH \
    --save_dir $SAVE_DIR

echo ">>> [2/2] RUNNING EVALUATION (TESTING)..."
# We deliberately use the config and model weights saved during training
# to guarantee 100% architectural parity.
python scripts/test.py \
    --config ${SAVE_DIR}/config.yaml \
    --checkpoint ${SAVE_DIR}/best_model.pth \
    --save_dir ${SAVE_DIR}/test_results

echo "======================================================="
echo "   PIPELINE COMPLETE. ALL RESULTS SAVED!               "
echo "======================================================="