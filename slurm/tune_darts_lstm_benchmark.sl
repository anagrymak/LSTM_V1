#!/usr/bin/env bash
#SBATCH --partition=gpu,hgx
#SBATCH --time=00-10:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=50GB
#SBATCH --output logs/%A_%a-%x.out
#SBATCH --error logs/%A_%a-%x.out
#SBATCH --array=0-9

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# select materials for training
MATERIALS="G2 G6 G4"

# tune the multiple sequence LSTM
RESULTS_DIR="results/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"
time python src/rnn_model.py tune \
    results/dataset_minutes.csv \
    "${RESULTS_DIR}" \
    --use-gpu \
    --n-trials 10 \
    --material $MATERIALS \
    --n-warmup 1000 2000 3000 \
    --warmup-max 1000 \
    --n-epochs-max 10 \
    --n-jobs 3 \
    --n-workers 9 \
    --memory-limit 4.5GiB \
    --float32-precision medium \
    --seed $SLURM_JOB_ID \
    --sampler random
