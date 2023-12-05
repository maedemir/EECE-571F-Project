#!/bin/bash
#SBATCH --job-name LUAD_20
#SBATCH --cpus-per-task 2
#SBATCH --mem 16000
#SBATCH --array=1-50
#SBATCH --output /home/zchen/test/aslide_20_%a.out
#SBATCH --error /home/zchen/test/aslide_20_%a.err
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_extract_annotated_patches_multi_magnification/
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=chenzha7@student.ubc.ca 
SINGULARITY_PATH=$(ls /opt/ | grep singularity | tail -1)
PATH="${PATH}:/opt/$SINGULARITY_PATH/bin"

singularity run -B /projects/ovcare/classification -B  /projects/AIM/TCGA -B /projects/AIM/ singularity_extract_annotated_patches.sif \
 from-arguments \
 --hd5_location /home/zchen/test/hd5/ \
 --num_patch_workers 1 \
 --store_thumbnail \
 use-directory \
 --slide_location /projects/ovcare/classification/Ali/TCGA_feature_extractor/patch_extraction/symlinks/diagnostics/LUAD/20/ \
 --patch_location /home/zchen/test/patch/ \
 --mask_location /projects/AIM/TCGA_maskes/LUAD/Diagnostic/ \
 --slide_pattern subtype \
 --slide_idx $SLURM_ARRAY_TASK_ID \
 --store_extracted_patches \
 use-entire-slide \
 --slide_coords_location /home/zchen/test/test.json \
 --patch_size 512\
 --magnifications 10 20 40 \
 --max_slide_patches 15\
