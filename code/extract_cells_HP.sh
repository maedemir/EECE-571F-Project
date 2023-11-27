#!/bin/bash

# Set absolute paths for directories and files
SOURCE_DIR="$( pwd -P )"
# IMG_DIR="$1"
# HOVERNET_OUTPUT_DIR="$2"
# OUTPUT_DIR="$3"

# Start running cell extraction code
for i in {001..050}; do
    python "code/extract_cells.py" \
        --image_dir "/projects/ovcare/classification/Ali/Heram/Dataset/Polyp_dataset/random_patches_10_20_40/Mix/HP/HP-$i/1000/20/" \
        --json_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output/HP/HP-$i/json" \
        --cell_image_patches_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output_extracted_cells/HP/HP-$i"
done
