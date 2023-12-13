#!/bin/bash
#SBATCH --job-name test_maedeh_TA
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=maeede.mir@gmail.com # Where to send mail 
#SBATCH --cpus-per-task 20
#SBATCH --mem=128000
#SBATCH --output /projects/ovcare/classification/Maedeh/%j.out
#SBATCH --error /projects/ovcare/classification/Maedeh/%j.out
#SBATCH -p gpu3090,gpuA6000,rtx5000,dgxV100
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
 
 
# Your program on dlhost
source /projects/ovcare/classification/Maedeh/py3env/bin/activate

for i in {001..063}; do
    python "/projects/ovcare/classification/Maedeh/EECE571F-project/hover_net/run_infer.py" \
        --model_mode=original \
        --model_path="/projects/ovcare/classification/Maedeh/EECE571F-project/hovernet_original_consep_notype_tf2pytorch.tar" \
        --batch_size=32 \
        --gpu=0 \
        tile \
        --input_dir="/projects/ovcare/classification/Ali/Heram/Dataset/Polyp_dataset/random_patches_10_20_40/Mix/TA/TA-$i/1000/20/" \
        --output="/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output/TA/TA-$i"
done