#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

for budget in 1 2.5 5 10; do
    for seed in 0 1 2 3 4; do
        budget_name="${budget/./p}"
        name="f3ed_initial_pool_only_${budget_name}pct_seed${seed}"

        python scripts/train_f3ed_f3set_tennis_first_round_only.py \
            --name "$name" \
            --seed "$seed" \
            --output_dir experiments/ \
            --initial_labeled_pool_size "$budget" \
            2>&1 | tee "logs/${name}.log"
    done
done