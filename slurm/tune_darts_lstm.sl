#!/usr/bin/env bash
#SBATCH --time=00-05:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=P100:1
#SBATCH --mem=5GB
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# create a folder for results
RESULTSDIR="results/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$RESULTSDIR"

# run the model fitting notebook
papermill -k ml_tooth \
    -p resultsdir "../$RESULTSDIR" \
    -p material G4 \
    -p sample S2 \
    -p train_ratio 0.4 \
    -p val_ratio 0.1 \
    -p n_trials 100 \
    -p n_epochs_max 500 \
    -p use_tpe True \
    --cwd notebooks \
    notebooks/03_tune_darts_lstm.ipynb \
    "${RESULTSDIR}/tune_darts_lstm.ipynb"
