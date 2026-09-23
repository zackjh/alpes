#!/bin/bash
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 SEED QUERY_STRATEGY QUERY_SCORE_POOLING"
    exit 1
fi

echo "=== Job information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo

echo "=== CPU ==="
lscpu | grep -E 'Model name|Socket|Core|Thread'

echo
echo "=== Memory ==="
echo "Memory allocated: ${SLURM_MEM_PER_NODE:-unknown} MB"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate alpes-f3set

cd "$HOME/alpes"

LOCAL_DIR="/tmp/$USER/$SLURM_JOB_ID"
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT

echo "=== Copying dataset archive ==="
echo "Start: $(date)"

cp "$HOME/alpes/data/f3set-tennis-frames.tar" "$LOCAL_DIR"

echo "End: $(date)"

echo
echo "=== Extracting dataset ==="
echo "Start: $(date)"

/usr/bin/time -v tar -xf "$LOCAL_DIR/f3set-tennis-frames.tar" -C "$LOCAL_DIR"

echo "End: $(date)"

initial_pool_size=10
query_batch_size=10
max_annotation_budget=100
seed="$1"
query_strategy="$2"
query_score_pooling="$3"
frame_dir="$LOCAL_DIR/data/f3set-tennis-frames"
output_dir="$HOME/alpes/experiments/4_active_learning_runs"

initial_pool_size_name="${initial_pool_size/./p}"
query_batch_size_name="${query_batch_size/./p}"
query_score_pooling_name="${query_score_pooling,,}"
query_strategy_name="${query_strategy,,}"
name="f3ed_${initial_pool_size_name}pct_${query_batch_size_name}pct_${query_strategy_name}_${query_score_pooling_name}pool_seed${seed}"

/usr/bin/time -v python scripts/train_f3ed_f3set_tennis.py \
    --name "$name" \
    --seed "$seed" \
    --frame_dir "$frame_dir" \
    --output_dir "$output_dir" \
    --initial_labeled_pool_size "$initial_pool_size" \
    --query_batch_size "$query_batch_size" \
    --max_annotation_budget "$max_annotation_budget" \
    --query_strategy "$query_strategy" \
    --query_score_pooling "$query_score_pooling"