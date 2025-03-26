#!/usr/bin/env bash
#SBATCH --time=00-10:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=P100:1
#SBATCH --mem=5GB
#SBATCH --output logs/%A_%a-%x.out
#SBATCH --error logs/%A_%a-%x.out
#SBATCH --array=0-14

# select material and sample
SAMPLE_LIST=($(tr -d '\r' < slurm/samples.csv))
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#SAMPLE_LIST[@]}" ]]; then
    echo "ERROR: accessing index ${SLURM_ARRAY_TASK_ID} in a list of ${#SAMPLE_LIST[@]} samples."
    exit 1
fi
TASK_SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}
MATERIAL=${TASK_SAMPLE%,*}
SAMPLE=${TASK_SAMPLE#*,}

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# create a folder for results
RESULTSDIR="results/${SLURM_ARRAY_JOB_ID}_${SLURM_JOB_NAME}/${MATERIAL}_${SAMPLE}"
mkdir -p "$RESULTSDIR"

# run the model fitting notebook
papermill -k ml_tooth \
    -p resultsdir "../$RESULTSDIR" \
    -p material $MATERIAL \
    -p sample $SAMPLE \
    -p train_ratio 0.4 \
    -p val_ratio 0.1 \
    -p n_trials 100 \
    -p n_epochs_max 500 \
    -p use_tpe True \
    --cwd notebooks \
    notebooks/03_tune_darts_lstm.ipynb \
    "${RESULTSDIR}/tune_darts_lstm.ipynb"
