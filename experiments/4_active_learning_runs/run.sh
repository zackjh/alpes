set -euo pipefail

mkdir -p logs

initial_pool_size=10
max_annotation_budget=100
for seed in 200; do
    for query_score_pooling in "MEAN"; do
        for query_batch_size in 10; do
            for query_strategy in "RANDOM_SAMPLING" "FINE_UNCERTAINTY_MEASURE" "FINE_ENTROPY_MEASURE" "GROUPED_FINE_ENTROPY" "COARSE_UNCERTAINTY_MEASURE" "COARSE_ENTROPY_MEASURE"; do
                initial_pool_size_name="${initial_pool_size/./p}"
                query_batch_size_name="${query_batch_size/./p}"
                query_score_pooling_name="${query_score_pooling,,}"
                query_strategy_name="${query_strategy,,}"
                name="f3ed_${initial_pool_size_name}pct_${query_batch_size_name}pct_${query_strategy_name}_${query_score_pooling_name}pool_seed${seed}"

                python scripts/train_f3ed_f3set_tennis.py \
                    --name "$name" \
                    --seed "$seed" \
                    --output_dir experiments/4_active_learning_runs \
                    --initial_labeled_pool_size "$initial_pool_size" \
                    --query_batch_size "$query_batch_size" \
                    --max_annotation_budget "$max_annotation_budget" \
                    --query_strategy "$query_strategy" \
                    --query_score_pooling "$query_score_pooling" \
                    2>&1 | tee "logs/${name}.log"
            done
        done
    done
done