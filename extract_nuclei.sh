#!/bin/bash

# Check if Git is available
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git and retry."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python and retry."
    exit 1
fi

# Clone from hover_net repository
git clone https://github.com/vqdang/hover_net.git

# Set absolute paths for directories and files
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
REPO_DIR="$SCRIPT_DIR/hover_net"
MODEL_PATH="$SCRIPT_DIR/hovernet_original_consep_notype_tf2pytorch.tar"

cd REPO_DIR

# Install Python dependencies
pip install -r "$REPO_DIR/requirements.txt"

# Quick fix: Install a specific version of docopt
pip install docopt==0.6.2

# Download a file using gdown
gdown https://drive.google.com/uc?id=1BF0GIgNGYpfyqEyU0jMsA6MqcUpVQx0b

# Run the Python script with error handling
python "$REPO_DIR/run_infer.py" \
    --model_mode=original \
    --model_path="$MODEL_PATH" \
    tile \
    --input_dir="imgs" \
    --output="." \
    --draw_dot \
    --save_qupath

# Check the return code of the Python script
if [ $? -ne 0 ]; then
    echo "Error: The Python script encountered a problem."
    exit 1
fi

echo "Script execution completed successfully."