#!/usr/bin/env bash

# 'set -e' makes the script stop if any command fails
set -e

# absolute dir of the script folder (Root directory of the project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================"
echo "       STARTING THE FULL FIFA RANKING PIPELINE        "
echo "======================================================"

echo -e "\n▶ [ STEP 0 ] Environment Setup..."

bash "$ROOT_DIR/cmd/environment_setup.sh"

source "$ROOT_DIR/.venv/bin/activate"

# ==============================================================================
# 1: DATA PIPELINE
# ==============================================================================
echo -e "\n▶ [ STEP 1 ] Executing Data Extraction and Refinement..."

# Executes the data pipeline script
bash "$ROOT_DIR/data/run_data_pipeline.sh"

if [ $? -ne 0 ]; then
    echo -e "\nERROR: Data pipeline failed. Aborting full execution."
    exit 1
fi

# ==============================================================================
# 2: MACHINE LEARNING PIPELINE
# ==============================================================================
echo -e "\n▶ [ STEP 2 ] Executing Model Training and Inference..."

# Executes the model training script
bash "$ROOT_DIR/ml_model_scripts/run_model_training.sh"

if [ $? -ne 0 ]; then
    echo -e "\nERROR: Model training failed. Aborting full execution."
    exit 1
fi

echo -e "\n======================================================"
echo "       FULL PIPELINE CONCLUDED SUCCESSFULLY!          "
echo "======================================================"