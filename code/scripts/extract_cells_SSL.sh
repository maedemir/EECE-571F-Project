#!/bin/bash
#SBATCH --job-name extract_cell_SSL
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=maeede.mir@gmail.com # Where to send mail
#SBATCH --cpus-per-task 20
#SBATCH --mem=128000
#SBATCH --output /projects/ovcare/classification/Maedeh/%j.out
#SBATCH --error /projects/ovcare/classification/Maedeh/%j.out

# Set absolute paths for directories and files
SOURCE_DIR="$( pwd -P )"
# IMG_DIR="$1"
# HOVERNET_OUTPUT_DIR="$2"
# OUTPUT_DIR="$3"


source /projects/ovcare/classification/Maedeh/py3env/bin/activate

# Start running cell extraction code
for i in {001..048}; do
    python "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/code/extract_cells.py" \
        --image_dir "/projects/ovcare/classification/Ali/Heram/Dataset/Polyp_dataset/random_patches_10_20_40/Mix/SSL/SSL-$i/1000/20/" \
        --json_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output/SSL/SSL-$i/json" \
        --cell_image_patches_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output_extracted_cells/SSL/SSL-$i"
done
