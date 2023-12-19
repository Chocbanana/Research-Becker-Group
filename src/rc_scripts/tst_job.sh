#!bin/bash
#SBATCH --nodes=1
#SBATCH --partition=aa100
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=test-job-bj
#SBATCH --output=test-job.%j.out

module purge
module load cuda/12.1.1
module load python/3.10.2

pip install --upgrade pip
pip install -r "../requirements.txt"

# Parameters to change
DATA_DIR="/projects/bhjo6995/data/gen_plasma_n64"
TARGET_DIR="/scratch/alpine/bhjo6995/data"
RUN_FOLDER="../notebooks/"
IPYNB_FILE="test_run_models.ipynb"

# Copy data files needed
./transfer_data.sh $DATA_DIR $TARGET_DIR

cd $RUN_FOLDER
jupyter nbconvert --to notebook --execute --allow-errors $IPYNB_FILE




