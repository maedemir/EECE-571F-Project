#!/bin/bash

# Set absolute paths for directories and files
SOURCE_DIR="$( pwd -P )"
IMG_DIR="$1"
OUTPUT_DIR="$2"

# Define the repositroy link and directory where you want to clone the repository
REPO_LINK="https://github.com/vqdang/hover_net.git"
REPO_DIR="$SOURCE_DIR/hover_net"

# Check if the directory already exists
if [ -d "$REPO_DIR" ]; then
    echo "The directory '$REPO_DIR' already exists. Skipping cloning."
else
    # Directory does not exist, so clone the repository
    git clone "$REPO_LINK"
    echo "Repository cloned to '$REPO_DIR'."
fi

cd REPO_DIR

# Download a fine-tuned model using gdown
gdown https://drive.google.com/uc?id=1BF0GIgNGYpfyqEyU0jMsA6MqcUpVQx0b

# Set Model path
MODEL_PATH="$SOURCE_DIR/hovernet_original_consep_notype_tf2pytorch.tar"


# Run the Python script with error handling
python "$REPO_DIR/run_infer.py" \
    --model_mode=original \
    --model_path="$MODEL_PATH" \
    tile \
    --input_dir="$IMG_DIR" \
    --output="$OUTPUT_DIR" 
    
# Check the return code of the Python script
if [ $? -ne 0 ]; then
    echo "Error: The Python script encountered a problem."
    exit 1
fi

echo "Script execution completed successfully."