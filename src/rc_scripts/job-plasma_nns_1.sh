#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --ntasks=16
#SBATCH --job-name=plasma_nns_1
#SBATCH --output=../../output/plasma_nns_1.%j.out

module purge
module load cuda/11.8
module load cudnn/8.6
module load ffmpeg/4.4
module load anaconda

# Parameters to change
DATA_DIR="/projects/${USER}/data/gen_plasma_n64"
TARGET_DIR="/scratch/alpine/${USER}/data"
RUN_FOLDER="../notebooks/"
IPYNB_FILE="run_models_RC"

# Copy data files needed
chmod a+x transfer_data.sh
./transfer_data.sh $DATA_DIR $TARGET_DIR

# If nnenv doesn't exist, create from rc_files/conda_env.yml
conda activate nnenv
# pip install --upgrade pip
# pip install -r "../requirements.txt"

cd $RUN_FOLDER
# Use Papermill (python lib) to run notebooks, since we then get stdout
papermill --autosave-cell-every=300 --log-output "${IPYNB_FILE}.ipynb" "../../output/${IPYNB_FILE}.out.ipynb"




