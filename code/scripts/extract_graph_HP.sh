#!/bin/bash
#SBATCH --job-name extract_graph_HP
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=maeede.mir@gmail.com # Where to send mail
#SBATCH --cpus-per-task 20
#SBATCH --mem=128000
#SBATCH --output /projects/ovcare/classification/Maedeh/%j.out
#SBATCH --error /projects/ovcare/classification/Maedeh/%j.out

# Set absolute paths for directories and files
SOURCE_DIR="$( pwd -P )"


source /projects/ovcare/classification/Maedeh/py3env/bin/activate

# Start running cell extraction code
for i in {001..050}; do
    python "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/code/extract_graph.py" \
        --json_dir_path "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output/HP/HP-$i/json" \
        --graph_output_dir "/projects/ovcare/classification/Maedeh/EECE571F-project/EECE-571F-Project/output_graph/HP/HP-$i" \
        --threshold 50 \
        --max_degree 10
done
