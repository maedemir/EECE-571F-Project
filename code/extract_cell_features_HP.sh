#!/bin/bash
#SBATCH --job-name test_maedeh_HP
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=maeede.mir@gmail.com # Where to send mail 
#SBATCH --cpus-per-task 20
#SBATCH --mem=128000
#SBATCH --output /projects/ovcare/classification/Maedeh/%j.out
#SBATCH --error /projects/ovcare/classification/Maedeh/%j.out
#SBATCH -p gpu3090,gpuA6000,rtx5000,dgxV100
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00

source /projects/ovcare/classification/Maedeh/py3env/bin/activate

for i in {001..050}; do
    python "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/code/extract_cell_features.py" \
        --features_output_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output_cell_features/HP" \
        --cell_image_patches_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output_extracted_cells/HP/HP-$i" \
        --class_name "HP"

done
