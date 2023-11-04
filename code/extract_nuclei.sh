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

# Define the directory where you want to clone the repository
REPO_LINK="https://github.com/vqdang/hover_net.git"

# Check if the directory already exists
if [ -d "$REPO_LINK" ]; then
    echo "The directory '$REPO_LINK' already exists. Skipping cloning."
else
    # Directory does not exist, so clone the repository
    git clone "$REPO_LINK"
    echo "Repository cloned to '$REPO_LINK'."
fi

# Set absolute paths for directories and files
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
REPO_DIR="$SCRIPT_DIR/hover_net"
MODEL_PATH="$SCRIPT_DIR/hovernet_original_consep_notype_tf2pytorch.tar"
IMG_DIR="$SCRIPT_DIR/imgs"

cd REPO_DIR

# Install Python dependencies
pip install -r "$REPO_DIR/requirements.txt"

# Quick fix: Install a specific version of docopt
pip install docopt==0.6.2

# Install gdown
pip install gdown

# Download a fine-tuned model using gdown
gdown https://drive.google.com/uc?id=1BF0GIgNGYpfyqEyU0jMsA6MqcUpVQx0b

# Run the Python script with error handling
python "$REPO_DIR/run_infer.py" \
    --model_mode=original \
    --model_path="$MODEL_PATH" \
    tile \
    --input_dir="$IMG_DIR" \
    --output="." \
    --draw_dot \
    --save_qupath

# Check the return code of the Python script
if [ $? -ne 0 ]; then
    echo "Error: The Python script encountered a problem."
    exit 1
fi

echo "Script execution completed successfully."