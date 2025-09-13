#!/bin/bash
#SBATCH --job-name=chairs-waftv2-dav2
#SBATCH --output=/n/fs/circuitnn/output_dir/chairs-waftv2-dav2.out
#SBATCH --chdir=/n/fs/circuitnn/WAFT/
#SBATCH --mail-user=yw7685@princeton.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=07-00:00:00
#SBATCH --gres=gpu:l40:8
seed=${1}
eval "$(conda shell.bash hook)"
conda activate flow
python -u train.py --cfg config/waft_v2/dav2/chairs.json --seed ${seed}