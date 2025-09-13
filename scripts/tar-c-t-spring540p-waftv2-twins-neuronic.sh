#!/bin/bash
#SBATCH --job-name=tar-c-t-spring-540p-waftv2-twins
#SBATCH --output=/n/fs/circuitnn/output_dir/tar-c-t-spring-540p-waftv2-twins.out
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
python -u train.py --cfg config/waft_v2/twins/tar-c-t-spring-540p.json --seed ${seed}