#!/bin/bash

#SBATCH --job-name=tier2-v5
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos

#SBATCH --gres=gpu:A6000:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=2-00:00:00

#SBATCH --requeue

#SBATCH --output=/home/apanda2/tier2-v5-%j.out
#SBATCH --error=/home/apanda2/tier2-v5-%j.err


set -euo pipefail


# =============================================================================
# ARGUMENTS
# =============================================================================

MODEL_FILTER="${1:?model filter required}"
OUT_DB="${2:?output database required}"


# =============================================================================
# ENVIRONMENT
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-/data/user_data/apanda2/reasoning_traces_data/multiturn_connections}"

ENV_DIR="/data/user_data/apanda2/conda/envs/vllm-cu128"

TIER1_DB="${PROJECT_DIR}/data/db/tier1_multiturn_all.db"

TIER2_MODEL="/data/user_data/apanda2/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"


cd "${PROJECT_DIR}"

# Use the source tree in the existing cluster environment.
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"


source /home/apanda2/miniforge3/etc/profile.d/conda.sh

conda activate "${ENV_DIR}"


export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


mkdir -p \
    "${PROJECT_DIR}/data/db/tier2_v5_full" \
    "${PROJECT_DIR}/logs"


# =============================================================================
# INFO
# =============================================================================

echo "============================================================"
echo "Tier-2 V5 full run"
echo "============================================================"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Started:      $(date --iso-8601=seconds)"

echo
echo "Trace model:  ${MODEL_FILTER}"
echo "Tier-1 DB:    ${TIER1_DB}"
echo "Output DB:    ${OUT_DB}"
echo "HF model:     ${TIER2_MODEL}"

echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "============================================================"

nvidia-smi -L


# =============================================================================
# RUN
#
# IMPORTANT:
# The script's tier2_done table makes this resumable.
# If this job is requeued/restarted with the same OUT_DB,
# already completed turns will be skipped.
# =============================================================================

python -m connections_research.extraction.stance \
    --tier1-db "${TIER1_DB}" \
    --out "${OUT_DB}" \
    --classifier hf \
    --model "${TIER2_MODEL}" \
    --model-filter "${MODEL_FILTER}" \
    --temperature 0.0 \
    --max-new-tokens 6 \
    --span-batch 16


echo
echo "============================================================"
echo "COMPLETED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
