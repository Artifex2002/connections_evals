#!/bin/bash
#SBATCH --job-name=nem8b-structured
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/nem8b-structured-%j.out
#SBATCH --error=logs/nem8b-structured-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail


PROJECT_DIR="${PROJECT_DIR:-/data/user_data/apanda2/reasoning_traces_data/multiturn_connections}"
ENV_DIR="/data/user_data/apanda2/conda/envs/vllm-cu128"

CHECKPOINT_MODEL="nvidia/Nemotron-H-8B-Reasoning-128K"
SERVED_MODEL="nemotron-h-8b-structured-memory"

BASE_URL="http://127.0.0.1:8000/v1"
DATABASE_PATH="${PROJECT_DIR}/data/db/connections.db"

CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/nemotron_h_8b_structured_seed42.json"

SERVER_LOG="${PROJECT_DIR}/logs/nem8b-structured-vllm-${SLURM_JOB_ID}.log"
BATCH_LOG="${PROJECT_DIR}/logs/nem8b-structured-batch-${SLURM_JOB_ID}.log"
BATCH_ERROR_LOG="${PROJECT_DIR}/logs/nem8b-structured-batch-${SLURM_JOB_ID}.err"


mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/checkpoints"

cd "${PROJECT_DIR}"

# Use the source tree in the existing cluster environment.
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

source /home/apanda2/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_DIR}"


export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="/data/user_data/${USER}/.hf_cache"
export HF_HUB_CACHE="/data/hf_cache/hub"


echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Physical model: ${CHECKPOINT_MODEL}"
echo "Experiment label: ${SERVED_MODEL}"
echo "Database: ${DATABASE_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "============================================================"

nvidia-smi


VLLM_PID=""
BATCH_PID=""


cleanup() {
    exit_code=$?

    echo
    echo "Cleanup started at $(date --iso-8601=seconds)"
    echo "Exit code: ${exit_code}"

    if [[ -n "${BATCH_PID}" ]] \
        && kill -0 "${BATCH_PID}" 2>/dev/null; then
        kill -TERM "${BATCH_PID}" 2>/dev/null || true
        wait "${BATCH_PID}" 2>/dev/null || true
    fi

    if [[ -n "${VLLM_PID}" ]] \
        && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    echo "Cleanup completed at $(date --iso-8601=seconds)"
    exit "${exit_code}"
}


checkpoint_and_requeue() {
    echo
    echo "Received USR1 at $(date --iso-8601=seconds)"
    echo "Requesting a clean batch stop before requeue."

    if [[ -n "${BATCH_PID}" ]] \
        && kill -0 "${BATCH_PID}" 2>/dev/null; then
        kill -USR1 "${BATCH_PID}" 2>/dev/null || true
    fi

    sleep 60

    echo "Requeuing Slurm job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
}


trap cleanup EXIT
trap checkpoint_and_requeue USR1


rm -f "${SERVER_LOG}"


vllm serve \
    "${CHECKPOINT_MODEL}" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for vLLM..."


while true; do
    if curl -sf \
        "http://127.0.0.1:8000/v1/models" \
        >/dev/null; then

        echo "vLLM is ready at $(date --iso-8601=seconds)"
        break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM crashed during startup."
        tail -n 200 "${SERVER_LOG}"
        exit 1
    fi

    echo "Still loading vLLM..."
    tail -n 3 "${SERVER_LOG}" || true
    sleep 15
done


curl -s \
    "http://127.0.0.1:8000/v1/models" \
    | python -m json.tool


python -m connections_research.runners.batch \
    --database "${DATABASE_PATH}" \
    --model "${SERVED_MODEL}" \
    --base-url "${BASE_URL}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --shuffle-seed 42 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 4096 \
    --max-model-turns 12 \
    --max-attempts-per-puzzle 2 \
    > >(tee -a "${BATCH_LOG}") \
    2> >(tee -a "${BATCH_ERROR_LOG}" >&2) &

BATCH_PID=$!

wait "${BATCH_PID}"
BATCH_EXIT_CODE=$?

echo
echo "Batch exited with code ${BATCH_EXIT_CODE}"
echo "Finished at $(date --iso-8601=seconds)"

exit "${BATCH_EXIT_CODE}"
