#!/bin/bash

#SBATCH --job-name=t2v5-shard
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos

#SBATCH --gres=gpu:A6000:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=2-00:00:00

#SBATCH --requeue

#SBATCH --output=/home/apanda2/t2v5-shard-%j.out
#SBATCH --error=/home/apanda2/t2v5-shard-%j.err


set -euo pipefail


MODEL_FILTER="${1:?model filter required}"
OUT_DB="${2:?output DB required}"
NUM_SHARDS="${3:?num shards required}"
SHARD_INDEX="${4:?shard index required}"


PROJECT_DIR="/data/user_data/apanda2/reasoning_traces_data/multiturn_connections"

ENV_DIR="/data/user_data/apanda2/conda/envs/vllm-cu128"

TIER1_DB="${PROJECT_DIR}/data/db/tier1_multiturn_all.db"

TIER2_MODEL="/data/user_data/apanda2/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"


cd "${PROJECT_DIR}"


source /home/apanda2/miniforge3/etc/profile.d/conda.sh

conda activate "${ENV_DIR}"


export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


echo "============================================================"
echo "Tier-2 V5 SHARD"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Started:      $(date --iso-8601=seconds)"
echo "Trace model:  ${MODEL_FILTER}"
echo "Output DB:    ${OUT_DB}"
echo "Shard:        ${SHARD_INDEX}/${NUM_SHARDS}"
echo "============================================================"

nvidia-smi -L


python tier2_scale_hf_multiturn.py \
    --tier1-db "${TIER1_DB}" \
    --out "${OUT_DB}" \
    --classifier hf \
    --model "${TIER2_MODEL}" \
    --model-filter "${MODEL_FILTER}" \
    --temperature 0.0 \
    --max-new-tokens 6 \
    --span-batch 16 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}"


echo
echo "============================================================"
echo "SHARD COMPLETED"
echo "Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
