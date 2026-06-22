#!/usr/bin/env bash

# 'set-e' makes the script stop if any command fails
set -e

# absolute dir of 'data' folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo "           STARTING FIFA DATA PIPELINE                "
echo "======================================================"

### =====================================================================================================================
# verifies if the validation file exists. if yes, ask before running the script again. if not, run the script directly
### =====================================================================================================================
run_step() {
    local step_title="$1"
    local python_script="$2"
    local validation_file="$3"

    echo -e "\n---> $step_title"

    if [ -f "$validation_file" ]; then
        echo "   [!] The data for this step already exists in the directory."
        # read -p ask a quesstion on the terminal. the common is N (no)
        read -p "   Would you like to run this script again? (y/N): " answer
        
        # if the user writes 'y' or 'Y', run the script
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
            echo "   Executando $python_script..."
            python "$python_script"
        else
            echo "   >> Pulando etapa (Cache mantido)."
        fi
    else
        # if the file doesn't exist, run the script directly without asking
        echo "   [!] Data not found. Executing script..."
        python "$python_script"
    fi
}

### ==============================================================================
# RUNNING THE STEPS
### ==============================================================================

# 1: EXTRACTION Wikipedia
run_step \
    "[1/4] Extracting External Metadata (Wikipedia)..." \
    "$SCRIPT_DIR/extract_external_metadata.py" \
    "$SCRIPT_DIR/external_metadata/world_teams_metadata.csv"

# 2: EXTRACTION FBref e FootyStats
run_step \
    "[2/4] Extracting Raw Data (World Cup/FBref)..." \
    "$SCRIPT_DIR/extract_data.py" \
    "$SCRIPT_DIR/raw/fbref_world_cup_games.csv"

# 3: PROCESSING AND ENRICHMENT
run_step \
    "[3/4] Processing Data, Clubs and Weights..." \
    "$SCRIPT_DIR/process_data.py" \
    "$SCRIPT_DIR/processed/processed_players.csv"

# 4: REFINEMENT AND FEATURE ENGINEERING
run_step \
    "[4/4] Calculating Efficiency and Synergy (Machine Learning)..." \
    "$SCRIPT_DIR/refine_data.py" \
    "$SCRIPT_DIR/refined/ml_national_synergy_features.csv"

echo -e "\n=========================================================="
echo "         DATA PIPENLINE CONCLUDED SUCCESSFULLY           "
echo "=========================================================="