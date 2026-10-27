#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

frame_dir="${1:-data/f3set-tennis-frames}"
checkpoint_root="experiments/1_determine_initial_pool"
status=0
found=0

while IFS= read -r -d '' checkpoint; do
    found=1
    if ! python scripts/val_f3ed_f3set_tennis.py \
        --checkpoint "$checkpoint" \
        --frame_dir "$frame_dir"; then
        status=1
    fi
done < <(find "$checkpoint_root" -type f -name 'checkpoint.pt' -print0 | sort -z)

if (( found == 0 )); then
    echo "No checkpoints found in $checkpoint_root" >&2
    status=1
fi

exit "$status"
