set -euo pipefail

mkdir -p logs

initial_pool_size=10
for query_batch_size in 5 15; do
    for seed in 0 1; do
        query_batch_size_name="${query_batch_size/./p}"
        name="optimized_${initial_pool_size}pct_${query_batch_size_name}pct_seed${seed}"

        python scripts/train_f3ed_f3set_tennis_optimized.py \
            --name "$name" \
            --seed "$seed" \
            --output_dir experiments/2_determine_query_batch_size \
            --initial_labeled_pool_size "$initial_pool_size" \
            --query_batch_size "$query_batch_size" \
            --query_strategy RANDOM_SAMPLING
            2>&1 | tee "logs/${name}.log"
    done
done