#!/usr/bin/env bash

# 'set -e' makes the script stop if any command fails
set -e

# absolute dir of the script folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# defines the Python executable from the virtual environment (adjust if your venv has a different name)
PYTHON_EXEC="$ROOT_DIR/.venv/bin/python"

# fallback: if the virtual environment is not found, use the default system python
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python"
fi

echo "======================================================"
echo "             STARTING MODEL PIPELINE                  "
echo "======================================================"

# ==============================================================================
# Helper function to run a step and log the progress
# ==============================================================================
run_step() {
    local step_message="$1"
    local script_path="$2"
    
    echo -e "\n------------------------------------------------------"
    echo -e "▶ $step_message"
    echo -e "------------------------------------------------------\n"
    
    # run python script
    "$PYTHON_EXEC" "$script_path"
    
    # verifies if the execution was succeded
    if [ $? -ne 0 ]; then
        echo -e "\nERROR: Step failed. Script execution aborted."
        exit 1
    fi
}

# ==============================================================================
# RUNNING THE STEPS
# ==============================================================================

# 1: Loads the best architecture (hyperparameters) for the model
run_step \
    "[1/3] Loading best architecture..." \
    "$SCRIPT_DIR/architecture_loader.py"

# 2: Train the model
run_step \
    "[2/3] Starting the model train..." \
    "$SCRIPT_DIR/train_model.py"

# 3: Generate the Ranking
run_step \
    "[3/3] Generating the National Teams Ranking..." \
    "$SCRIPT_DIR/generate_fifa_ranking.py"

echo -e "\n=========================================================="
echo "           MODEL PIPELINE CONCLUDED SUCCESSFULLY             "
echo "=========================================================="