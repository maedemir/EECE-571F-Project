#!/bin/bash

# Set absolute paths for directories and files
SOURCE_DIR="$( pwd -P )"
IMG_DIR="$1"
HOVERNET_OUTPUT_DIR="$2"
OUTPUT_DIR="$3"

# Start running cell extraction code
python "code/extract_cells.py" \
    --image-dir = "$IMG_DIR" \
    --json-dir = "$HOVERNET_OUTPUT_DIR" \
    --cell-image-patches-dir = "$OUTPUT_DIR"
    

