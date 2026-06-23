#!/usr/bin/env bash

# 'set -e' makes the script stop if any command fails
set -e

echo "======================================================"
echo "       AUTOMATIC ENVIRONMENT SETUP (UV)               "
echo "======================================================"

# 1. verifies if the 'uv' is already installed at the system
if ! command -v uv &> /dev/null; then
    echo -e "\n▶ 'uv' not found. Downloading and installing astral-sh/uv..."
    
    # executes the official install
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # add the basic dir to the uv on PATH to this terminal section
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo -e "▶ 'uv' successfully installed!"
else
    echo -e "\n▶ 'uv' is already installed."
fi

# show the version to confirm
echo -n "▶ uv version: "
uv --version

# 2. syncronize the environment
echo -e "\n▶ Syncing Python environment and installing dependencies..."
uv sync

echo -e "\n======================================================"
echo "       SETUP COMPLETE! YOU ARE READY TO GO!           "
echo "======================================================"