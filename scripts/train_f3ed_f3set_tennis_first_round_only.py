"""Train F3ED once on an initial labeled F3Set-Tennis pool.

This is the first-round-only counterpart of ``train_f3ed_f3set_tennis.py``.
It intentionally reuses that script's pool construction and training routine so
experiments which vary ``--initial_labeled_pool_size`` remain directly
comparable to round zero of the active-learning experiment.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from src.F3Set.util.dataset import load_classes
from torch.backends import cudnn

from train_f3ed_f3set_tennis import (
    BATCH_SIZE,
    CLIP_LEN,
    CROP_DIM,
    EPOCH_NUM_FRAMES,
    LEARNING_RATE,
    NUM_EPOCHS,
    START_VAL_EPOCH,
    STRIDE,
    WARM_UP_EPOCHS,
    WINDOW,
    frame_budget_from_percentage,
    random_clips_to_frame_budget,
    save_json,
    train_round,
)

# Use deterministic settings for reproducibility
cudnn.benchmark = False
cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train F3ED for one round on an initial labeled F3Set-Tennis pool"
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--initial_labeled_pool_size",
        type=float,
        required=True,
        metavar="PERCENT",
        help="initial labeled-frame budget as a percentage of the training set",
    )
    return parser.parse_args()


def seed_experiment(seed: int) -> None:
    """Seed every RNG used by pool selection and F3ED training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    if not 0 < args.initial_labeled_pool_size <= 100:
        raise ValueError("initial labeled pool size must be in (0, 100] percent")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "src" / "F3Set" / "data" / "f3set-tennis"
    frame_dir = project_root / "data" / "f3set-tennis-frames"

    train_file = dataset_root / "train.json"
    val_file = dataset_root / "val.json"
    test_file = dataset_root / "test.json"

    run_dir = args.output_dir.expanduser().resolve() / args.name
    round_dir = run_dir / "round_000"
    round_dir.mkdir(parents=True, exist_ok=False)

    with train_file.open() as f:
        train_annotations = json.load(f)

    frame_counts = [annotation["num_frames"] for annotation in train_annotations]
    total_training_frames = sum(frame_counts)
    initial_frame_budget = frame_budget_from_percentage(
        args.initial_labeled_pool_size,
        total_training_frames,
    )

    # Use a dedicated RNG for pool selection. This prevents selection from
    # advancing the RNG state used for model initialization and augmentation.
    seed_experiment(args.seed)
    pool_rng = random.Random(args.seed)
    labeled_indices = set(
        random_clips_to_frame_budget(
            set(range(len(train_annotations))),
            frame_counts,
            initial_frame_budget,
            pool_rng,
        )
    )

    labeled_frames = sum(frame_counts[index] for index in labeled_indices)
    labeled_file = round_dir / "labeled_train.json"
    save_json(
        labeled_file,
        [train_annotations[index] for index in sorted(labeled_indices)],
    )

    save_json(
        run_dir / "config.json",
        {
            "name": args.name,
            "seed": args.seed,
            "initial_labeled_pool_size": args.initial_labeled_pool_size,
            "active_learning_budget_unit": "percent_of_training_frames",
            "total_training_frames": total_training_frames,
            "initial_labeled_pool_frame_budget": initial_frame_budget,
            "first_round_only": True,
            "f3ed": {
                "feature_arch": "rny002",
                "temporal_arch": "gru",
                "use_ctx": True,
                "epoch_num_frames": EPOCH_NUM_FRAMES,
                "clip_len": CLIP_LEN,
                "stride": STRIDE,
                "crop_dim": CROP_DIM,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "warm_up_epochs": WARM_UP_EPOCHS,
                "start_val_epoch": START_VAL_EPOCH,
                "learning_rate": LEARNING_RATE,
                "window": WINDOW,
            },
        },
    )

    print(
        "\n=== Round 0 (initial labeled pool only) ===\n"
        f"Labeled: {len(labeled_indices)} clips / {labeled_frames} frames\n"
        f"Budget:  {args.initial_labeled_pool_size:g}% "
        f"({initial_frame_budget} target frames)"
    )

    classes = load_classes(str(dataset_root / "elements.txt"))
    _, best_epoch, best_val_edit, test_edit = train_round(
        classes,
        labeled_file,
        val_file,
        test_file,
        frame_dir,
        round_dir,
    )

    save_json(
        run_dir / "history.json",
        [
            {
                "round": 0,
                "labeled_pool_size": len(labeled_indices),
                "labeled_pool_frames": labeled_frames,
                "labeled_budget_percent": (
                    100 * labeled_frames / total_training_frames
                ),
                "best_epoch": best_epoch,
                "val_edit": best_val_edit,
                "test_edit": test_edit,
            }
        ],
    )


if __name__ == "__main__":
    main()
