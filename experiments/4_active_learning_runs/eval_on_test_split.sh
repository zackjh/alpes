#!/usr/bin/env bash
set -euo pipefail

frame_dir=data/f3set-tennis-frames

runs=(
    f3ed_10pct_10pct_weighted_uncertainty_measure_meanpool_seed200
    f3ed_10pct_10pct_weighted_uncertainty_measure_meanpool_seed400
    f3ed_10pct_10pct_weighted_uncertainty_measure_meanpool_seed500
    f3ed_10pct_10pct_weighted_entropy_measure_meanpool_seed100
    f3ed_10pct_10pct_weighted_entropy_measure_meanpool_seed200
    f3ed_10pct_10pct_weighted_entropy_measure_meanpool_seed400
    f3ed_10pct_10pct_weighted_entropy_measure_meanpool_seed500
)

status=0
for run in "${runs[@]}"; do
    run_dir="experiments/4_active_learning_runs/$run"
    checkpoints=("$run_dir"/round_*/checkpoint.pt)
    if (( ${#checkpoints[@]} == 0 )); then
        echo "No AL round checkpoints found in $run_dir" >&2
        status=1
        continue
    fi

    for checkpoint in "${checkpoints[@]}"; do
        round_dir="$(dirname "$checkpoint")"
        echo "Evaluating $checkpoint"
        python scripts/test_f3ed_f3set_tennis.py \
            --checkpoint "$checkpoint" \
            --frame_dir "$frame_dir" \
            --output "$round_dir/test_results.json"
    done
done

exit "$status"
