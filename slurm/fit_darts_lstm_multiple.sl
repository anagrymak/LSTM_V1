#!/usr/bin/env bash
#SBATCH --partition=gpu,hgx
#SBATCH --time=00-05:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=5GB
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# select materials for training
MATERIALS="G5 G6 G8"

# create results folder
RESULTS_DIR="results/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$RESULTS_DIR"

# train model
python src/rnn_model.py fit \
    results/dataset_minutes.csv \
    "${RESULTS_DIR}" \
    --use-gpu \
    --verbosity 1 \
    --training-materials $MATERIALS \
    --warmup 1000 \
    --n-epochs 200 \
    --lr 0.004 \
    --batch-size 128 \
    --n-rnn-layers 2 \
    --hidden-dim 16 \
    --gradient-clip 0.1 \
    --no-positive

# evaluate on training data
python src/rnn_model.py evaluate \
    results/dataset_minutes.csv \
    "${RESULTS_DIR}" \
    "${RESULTS_DIR}" \
    --n-warmup 1000 \
    --materials G7 $MATERIALS \
    --no-positive
