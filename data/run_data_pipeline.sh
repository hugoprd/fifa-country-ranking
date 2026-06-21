#!/usr/bin/env bash

#########################
### RUN DATA PIPELINE ###
#########################

# ==============================================================================
# Critical Safety Barrier (Fail-Fast):
# the 'set -e' makes the bash script stop immediately if any 
# python script returns an error. this prevents the domino effect of failures.
# ==============================================================================
set -e

# discover the absolute directory where this bash script is saved.
# this ensures that you can run the command from any location in the terminal
# and it will always find the correct Python files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================================="
echo "        STARTING DATA PIPELINE FOR FIFA COUNTRY RANKING        "
echo "==============================================================="

echo -e "\n---> [1/4] Extracting External Metadata (Wikipedia)..."
python "$SCRIPT_DIR/extract_external_metadata.py"

echo -e "\n---> [2/4] Extracting Raw Data (Club World Cup)..."
python "$SCRIPT_DIR/extract_data.py"

echo -e "\n---> [3/4] Processing Data and Calculating Weights..."
python "$SCRIPT_DIR/process_data.py"

echo -e "\n---> [4/4] Refining Final Data..."
python "$SCRIPT_DIR/refine_data.py"

echo -e "\n===================================================================="
echo "                 PIPELINE CONCLUDED SUCCESSFULLY.                  "
echo "===================================================================="