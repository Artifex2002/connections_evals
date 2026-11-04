#!/bin/bash
#SBATCH --job-name=nemotron-connections
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/nemotron-connections-%j.out
#SBATCH --error=logs/nemotron-connections-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail


PROJECT_DIR="${PROJECT_DIR:-/data/user_data/apanda2/reasoning_traces_data/multiturn_connections}"
ENV_DIR="/data/user_data/apanda2/conda/envs/vllm-cu128"

MODEL_NAME="nvidia/Nemotron-H-8B-Reasoning-128K"
BASE_URL="http://127.0.0.1:8000/v1"

DATABASE_PATH="${PROJECT_DIR}/data/db/connections.db"

CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/nemotron_all_seed42.json"

SERVER_LOG="${PROJECT_DIR}/logs/nemotron-vllm-${SLURM_JOB_ID}.log"

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

# Shared read-only model cache where available.
export HF_HUB_CACHE="/data/hf_cache/hub"

# User-writable Hugging Face cache and metadata.
export HF_HOME="/data/user_data/${USER}/.hf_cache"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"

nvidia-smi


VLLM_PID=""
BATCH_PID=""


cleanup() {
    exit_code=$?

    echo
    echo "Cleanup started: $(date --iso-8601=seconds)"
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

    echo "Cleanup completed: $(date --iso-8601=seconds)"
    exit "${exit_code}"
}


checkpoint_and_requeue() {
    echo
    echo "Received USR1 at $(date --iso-8601=seconds)"
    echo "Stopping the batch runner after its current state is saved."

    if [[ -n "${BATCH_PID}" ]] \
        && kill -0 "${BATCH_PID}" 2>/dev/null; then
        kill -USR1 "${BATCH_PID}" 2>/dev/null || true
    fi

    # Give the Python runner time to finish or persist the current
    # puzzle-level checkpoint.
    sleep 30

    echo "Requesting requeue for job ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
}


trap cleanup EXIT
trap checkpoint_and_requeue USR1


rm -f "${SERVER_LOG}"

vllm serve \
    "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for vLLM readiness..."


until curl -sf \
    "http://127.0.0.1:8000/v1/models" \
    >/dev/null; do

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM crashed during startup."
        tail -n 200 "${SERVER_LOG}"
        exit 1
    fi

    sleep 10
done


echo "vLLM is ready: $(date --iso-8601=seconds)"

curl -s \
    "http://127.0.0.1:8000/v1/models" \
    | python -m json.tool


python -m connections_research.runners.batch \
    --database "${DATABASE_PATH}" \
    --model "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --shuffle-seed 42 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 4096 \
    --max-model-turns 12 \
    --max-attempts-per-puzzle 2 \
    > >(tee -a "${PROJECT_DIR}/logs/nemotron-batch-${SLURM_JOB_ID}.log") \
    2> >(tee -a "${PROJECT_DIR}/logs/nemotron-batch-${SLURM_JOB_ID}.err" >&2) &

BATCH_PID=$!

wait "${BATCH_PID}"
BATCH_EXIT_CODE=$?

echo "Batch exit code: ${BATCH_EXIT_CODE}"
echo "Finished: $(date --iso-8601=seconds)"

exit "${BATCH_EXIT_CODE}"
