#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

SYSTEM_PROMPT_PREFIX="${SYSTEM_PROMPT_PREFIX:-}"

PROJECT_DIR="${PROJECT_DIR:-/data/user_data/apanda2/reasoning_traces_data/multiturn_connections}"
ENV_DIR="/data/user_data/apanda2/conda/envs/vllm-cu128"

: "${MODEL_ID:?MODEL_ID is required}"
: "${SERVED_MODEL:?SERVED_MODEL is required}"
: "${DATABASE_PATH:?DATABASE_PATH is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"

TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_TURNS="${MAX_MODEL_TURNS:-12}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
LIMIT="${LIMIT:-}"

# Give every Slurm job its own localhost port so multiple vLLM
# servers on the same physical node cannot collide.
VLLM_PORT="${VLLM_PORT:-$((18000 + SLURM_JOB_ID % 10000))}"
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/checkpoints" \
    "${PROJECT_DIR}/data/db/models"

cd "${PROJECT_DIR}"

# Use the source tree in the existing cluster environment.
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

source /home/apanda2/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use Ash's personal model cache, not Babel's shared cache.
export HF_HOME="/data/user_data/apanda2/.hf_cache"
export HF_HUB_CACHE="/data/user_data/apanda2/.hf_cache/hub"

# Fail instead of downloading unexpectedly during a compute job.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


SERVER_LOG="${PROJECT_DIR}/logs/${SERVED_MODEL}-vllm-${SLURM_JOB_ID}.log"
BATCH_LOG="${PROJECT_DIR}/logs/${SERVED_MODEL}-batch-${SLURM_JOB_ID}.log"
BATCH_ERR="${PROJECT_DIR}/logs/${SERVED_MODEL}-batch-${SLURM_JOB_ID}.err"

VLLM_PID=""
BATCH_PID=""


cleanup() {
    exit_code=$?

    echo
    echo "Cleanup at $(date --iso-8601=seconds)"
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

    exit "${exit_code}"
}


checkpoint_and_requeue() {
    echo
    echo "Received USR1 at $(date --iso-8601=seconds)"

    if [[ -n "${BATCH_PID}" ]] \
        && kill -0 "${BATCH_PID}" 2>/dev/null; then
        kill -USR1 "${BATCH_PID}" 2>/dev/null || true
    fi

    sleep 60

    echo "Requeuing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
}


trap cleanup EXIT
trap checkpoint_and_requeue USR1


echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Model checkpoint: ${MODEL_ID}"
echo "Experiment label: ${SERVED_MODEL}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "vLLM port: ${VLLM_PORT}"
echo "Base URL: ${BASE_URL}"
echo "Database: ${DATABASE_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Max attempts per puzzle: ${MAX_ATTEMPTS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

nvidia-smi -L
nvidia-smi


VLLM_ARGS=(
    serve
    "${MODEL_ID}"
    --served-model-name "${SERVED_MODEL}"
    --host 127.0.0.1
    --port "${VLLM_PORT}"
    --dtype bfloat16
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization 0.90
)

# if [[ "${TP_SIZE}" -gt 1 ]]; then
#     VLLM_ARGS+=(
#         --distributed-executor-backend mp
#     )
# fi


rm -f "${SERVER_LOG}"

vllm "${VLLM_ARGS[@]}" \
    > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!


####################

echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for ${SERVED_MODEL} on ${BASE_URL}..."

VLLM_READY=0

for i in $(seq 1 180); do
    MODELS_JSON="$(
        curl -sf "${BASE_URL}/models" 2>/dev/null || true
    )"

    # Do not merely check whether a server answered.
    # Verify that it is serving OUR requested model.
    if [[ -n "${MODELS_JSON}" ]] && \
       echo "${MODELS_JSON}" | \
       python -c '
import json
import sys

expected = sys.argv[1]

try:
    payload = json.load(sys.stdin)
    ids = [
        model.get("id")
        for model in payload.get("data", [])
    ]
except Exception:
    sys.exit(1)

sys.exit(0 if expected in ids else 1)
' "${SERVED_MODEL}"
    then
        echo
        echo "vLLM is ready at $(date --iso-8601=seconds)"
        echo "Verified served model: ${SERVED_MODEL}"
        VLLM_READY=1
        break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo
        echo "vLLM crashed during startup."
        tail -n 250 "${SERVER_LOG}"
        exit 1
    fi

    echo "Still loading ${SERVED_MODEL}... attempt ${i}/180"
    tail -n 3 "${SERVER_LOG}" || true
    sleep 10
done

if [[ "${VLLM_READY}" -ne 1 ]]; then
    echo
    echo "Timed out waiting for ${SERVED_MODEL}."
    echo "Last /models response:"
    echo "${MODELS_JSON:-<none>}"
    echo
    echo "Last 250 vLLM log lines:"
    tail -n 250 "${SERVER_LOG}" || true
    exit 1
fi

echo
echo "vLLM /models:"
curl -s "${BASE_URL}/models" | python -m json.tool

##########################

BATCH_ARGS=(
    -m connections_research.runners.batch
    --database "${DATABASE_PATH}"
    --model "${SERVED_MODEL}"
    --base-url "${BASE_URL}"
    --checkpoint "${CHECKPOINT_PATH}"
    --shuffle-seed 42
    --temperature 0.6
    --top-p 0.95
    --max-tokens "${MAX_TOKENS}"
    --max-model-turns "${MAX_MODEL_TURNS}"
    --max-attempts-per-puzzle "${MAX_ATTEMPTS}"
)

if [[ -n "${SYSTEM_PROMPT_PREFIX}" ]]; then
    BATCH_ARGS+=(
        --system-prompt-prefix "${SYSTEM_PROMPT_PREFIX}"
    )
fi

if [[ -n "${LIMIT}" ]]; then
    BATCH_ARGS+=(
        --limit "${LIMIT}"
    )
fi


python "${BATCH_ARGS[@]}" \
    > >(tee -a "${BATCH_LOG}") \
    2> >(tee -a "${BATCH_ERR}" >&2) &

BATCH_PID=$!

wait "${BATCH_PID}"
BATCH_EXIT_CODE=$?

echo
echo "Batch exit code: ${BATCH_EXIT_CODE}"
echo "Finished: $(date --iso-8601=seconds)"

exit "${BATCH_EXIT_CODE}"
