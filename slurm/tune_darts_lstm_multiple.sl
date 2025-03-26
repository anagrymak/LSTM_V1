#!/usr/bin/env bash
#SBATCH --partition=gpu,hgx
#SBATCH --time=00-10:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=35GB
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# add src/ to PYTHONPATH (not great but allow dask workers to access code)
export PYTHONPATH="$PYTHONPATH:src"

# select materials for training
MATERIALS="G5 G6 G8 G9 G10 G11"

# create results folder
RESULTS_DIR="results/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$RESULTS_DIR"

# disable Dask worker memory management, except killing worker as last resort
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95

# start dask scheduler
SCHEDULER_FILE="${RESULTS_DIR}/sheduler.json"
SCHEDULER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
dask scheduler --scheduler-file "$SCHEDULER_FILE" --port "$SCHEDULER_PORT" --interface ib0 &

# wait 20s for the scheduler file to exist
sleep 20
if [[ ! -f "$SCHEDULER_FILE" ]]; then
    echo "Missing scheduler file "${SCHEDULER_FILE}"!"
    exit 1
fi

# start workers on each GPU
for i in ${CUDA_VISIBLE_DEVICES//,/ }; do
    CUDA_VISIBLE_DEVICES=$i dask worker \
        --nthreads 1 \
        --nworkers 6 \
        --memory-limit 5GB \
        --local-directory "$RESULTS_DIR" \
        --scheduler-file "$SCHEDULER_FILE" &
done

# tune the multiple sequence LSTM
python src/rnn_model.py tune \
    results/dataset_minutes_10m.csv \
    "${RESULTS_DIR}" \
    --use-gpu \
    --n-trials 200 \
    --material $MATERIALS \
    --n-warmup 100 \
    --warmup-max 100 \
    --n-epochs-max 500 \
    --n-jobs 1 \
    --scheduler-file "$SCHEDULER_FILE" \
    --sampler tpe \
    --metric MAE_avg \
    --training-length 400 \
    --warmup 100 \
    --batch-size 32 \
    --n-rnn-layers 2

# retrain a model using all materials
python src/rnn_model.py fit \
    results/dataset_minutes_10m.csv \
    "${RESULTS_DIR}/final_model" \
    --config "${RESULTS_DIR}/best_hyperparams.json" \
    --use-gpu \
    --verbosity 1 \
    --training-materials $MATERIALS \
    --training-length 400 \
    --warmup 100 \
    --batch-size 32 \
    --n-rnn-layers 2
