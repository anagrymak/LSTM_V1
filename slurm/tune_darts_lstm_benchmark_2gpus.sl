#!/usr/bin/env bash
#SBATCH --partition=gpu,hgx
#SBATCH --time=00-10:00:00
#SBATCH --account=uoo03699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=A100:2
#SBATCH --mem=50GB
#SBATCH --output logs/%A_%a-%x.out
#SBATCH --error logs/%A_%a-%x.out
#SBATCH --array=0-39

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

# activate Python virtual environment
module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# add src/ to PYTHONPATH (not great but allow dask workers to access code)
export PYTHONPATH="$PYTHONPATH:src"

# select materials for training
MATERIALS="G2 G6 G4"

# create results folder
RESULTS_DIR="results/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"
mkdir -p "$RESULTS_DIR"

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
        --nworkers 3 \
        --memory-limit 4.5GiB \
        --local-directory "$RESULTS_DIR" \
        --scheduler-file "$SCHEDULER_FILE" &
done

# tune the multiple sequence LSTM
time python src/rnn_model.py tune \
    results/dataset_minutes.csv \
    "${RESULTS_DIR}" \
    --use-gpu \
    --n-trials 10 \
    --material $MATERIALS \
    --n-warmup 1000 2000 3000 \
    --warmup-max 1000 \
    --n-epochs-max 10 \
    --n-jobs 2 \
    --scheduler-file "$SCHEDULER_FILE" \
    --float32-precision medium \
    --seed $SLURM_JOB_ID \
    --sampler random
