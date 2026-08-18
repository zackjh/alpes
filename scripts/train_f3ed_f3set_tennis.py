"""Train F3ED with pool-based active learning on F3Set-Tennis."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from src.F3Set.dataset.frame_process import (
    ActionSeqDataset,
    ActionSeqVideoDataset,
)
from src.F3Set.train_f3set_f3ed import F3Set, evaluate
from src.F3Set.util.dataset import load_classes
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

# F3ED training configuration from the original F3Set implementation.
EPOCH_NUM_FRAMES = 500_000
CLIP_LEN = 96
STRIDE = 2
CROP_DIM = 224
BATCH_SIZE = 4
NUM_EPOCHS = 50
WARM_UP_EPOCHS = 3
START_VAL_EPOCH = 30
LEARNING_RATE = 0.001
WINDOW = 5
BASE_NUM_WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pool-based active learning for F3ED on F3Set-Tennis"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--initial_labeled_pool_size", type=int, required=True)
    parser.add_argument("--query_batch_size", type=int, required=True)
    parser.add_argument(
        "--query_strategy",
        required=True,
        choices=[
            "RANDOM_SAMPLING",
            "UNCERTAINTY_MEASURE",
            "ENTROPY_MEASURE",
        ],
    )
    return parser.parse_args()


def save_json(path: Path, data: object) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def query(
    strategy: str,
    model: F3Set,
    unlabeled_indices: set[int],
    query_size: int,
    rng: random.Random,
) -> list[int]:
    """Select samples from the unlabeled pool."""
    candidates = sorted(unlabeled_indices)
    query_size = min(query_size, len(candidates))

    if strategy == "RANDOM_SAMPLING":
        return rng.sample(candidates, query_size)

    if strategy == "UNCERTAINTY_MEASURE":
        raise NotImplementedError("Uncertainty sampling is not implemented yet")

    if strategy == "ENTROPY_MEASURE":
        raise NotImplementedError("Entropy sampling is not implemented yet")

    raise ValueError(f"Unknown query strategy: {strategy}")


def train_round(
    classes: dict,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    frame_dir: Path,
    round_dir: Path,
) -> tuple[F3Set, int, float, float | None]:
    """Train F3ED from scratch on the current labeled pool."""
    dataset_len = EPOCH_NUM_FRAMES // (CLIP_LEN * STRIDE)

    train_data = ActionSeqDataset(
        classes,
        str(train_file),
        str(frame_dir),
        CLIP_LEN,
        dataset_len,
        is_eval=False,
        crop_dim=CROP_DIM,
        stride=STRIDE,
    )
    val_data = ActionSeqDataset(
        classes,
        str(val_file),
        str(frame_dir),
        CLIP_LEN,
        dataset_len // 4,
        is_eval=True,
        crop_dim=CROP_DIM,
        stride=STRIDE,
    )
    val_video_data = ActionSeqVideoDataset(
        classes,
        str(val_file),
        str(frame_dir),
        CLIP_LEN,
        crop_dim=CROP_DIM,
        stride=STRIDE,
        overlap_len=0,
    )

    train_data.print_info()
    val_data.print_info()

    # This matches F3Set's epoch-dependent DataLoader worker seeding.
    epoch = 0

    def worker_init_fn(worker_id: int) -> None:
        random.seed(worker_id + epoch * 100)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=min(os.cpu_count() or 1, BASE_NUM_WORKERS * 2),
        prefetch_factor=1,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn,
    )

    model = F3Set(
        len(classes),
        "rny002",
        "gru",
        CLIP_LEN,
        step=STRIDE,
        window=WINDOW,
        use_ctx=True,
        device="cuda",
        multi_gpu=False,
    )

    optimizer, scaler = model.get_optimizer({"lr": LEARNING_RATE})

    steps_per_epoch = len(train_loader)
    scheduler = ChainedScheduler(
        [
            LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=WARM_UP_EPOCHS * steps_per_epoch,
            ),
            CosineAnnealingLR(
                optimizer,
                (NUM_EPOCHS - WARM_UP_EPOCHS) * steps_per_epoch,
            ),
        ]
    )

    losses = []
    best_epoch = None
    best_val_edit = -math.inf
    checkpoint_file = round_dir / "checkpoint.pt"

    for epoch in range(NUM_EPOCHS):
        train_loss = model.epoch(
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=scheduler,
            epoch=epoch,
        )
        val_loss = model.epoch(
            val_loader,
            epoch=epoch,
        )

        val_edit = 0.0

        if epoch >= START_VAL_EPOCH:
            _, _, val_edit = evaluate(
                model,
                val_video_data,
                classes,
                window=WINDOW,
            )

            if val_edit > best_val_edit:
                best_val_edit = val_edit
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_file)

        losses.append(
            {
                "epoch": epoch,
                "train": train_loss,
                "val": val_loss,
                "val_edit": val_edit,
            }
        )
        save_json(round_dir / "loss.json", losses)

        print(
            f"[Epoch {epoch}] "
            f"Train loss: {train_loss:.5f} "
            f"Val loss: {val_loss:.5f} "
            f"Val edit: {val_edit:.5f}"
        )

    assert best_epoch is not None

    model.load(torch.load(checkpoint_file))

    test_edit = None

    if test_file.exists():
        test_data = ActionSeqVideoDataset(
            classes,
            str(test_file),
            str(frame_dir),
            CLIP_LEN,
            crop_dim=CROP_DIM,
            stride=STRIDE,
            overlap_len=CLIP_LEN // 2,
        )
        _, _, test_edit = evaluate(
            model,
            test_data,
            classes,
            window=WINDOW,
        )

    return model, best_epoch, best_val_edit, test_edit


def main() -> None:
    args = parse_args()

    if args.initial_labeled_pool_size <= 0:
        raise ValueError("initial labeled pool size must be positive")

    if args.query_batch_size <= 0:
        raise ValueError("query batch size must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "src" / "F3Set" / "data" / "f3set-tennis"
    frame_dir = project_root / "data" / "f3set-tennis-frames"

    train_file = dataset_root / "train.json"
    val_file = dataset_root / "val.json"
    test_file = dataset_root / "test.json"

    run_dir = args.output_dir.expanduser().resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    with train_file.open() as f:
        train_annotations = json.load(f)

    if args.initial_labeled_pool_size > len(train_annotations):
        raise ValueError(
            "initial labeled pool size cannot exceed the training set size"
        )

    # Seed the experiment while otherwise leaving F3ED's training behaviour
    # unchanged.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rng = random.Random(args.seed)

    all_indices = list(range(len(train_annotations)))
    labeled_indices = set(rng.sample(all_indices, args.initial_labeled_pool_size))
    unlabeled_indices = set(all_indices) - labeled_indices

    classes = load_classes(str(dataset_root / "elements.txt"))

    save_json(
        run_dir / "config.json",
        {
            "name": args.name,
            "seed": args.seed,
            "query_strategy": args.query_strategy,
            "initial_labeled_pool_size": args.initial_labeled_pool_size,
            "query_batch_size": args.query_batch_size,
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

    history = []
    round_number = 0

    while True:
        round_dir = run_dir / f"round_{round_number:03d}"
        round_dir.mkdir()

        print(
            f"\n=== Round {round_number} ===\n"
            f"Labeled:   {len(labeled_indices)}\n"
            f"Unlabeled: {len(unlabeled_indices)}"
        )

        # ActionSeqDataset expects an annotation JSON, so expose the current
        # labeled pool as one.
        labeled_file = round_dir / "labeled_train.json"
        save_json(
            labeled_file,
            [train_annotations[i] for i in sorted(labeled_indices)],
        )

        model, best_epoch, best_val_edit, test_edit = train_round(
            classes,
            labeled_file,
            val_file,
            test_file,
            frame_dir,
            round_dir,
        )

        history.append(
            {
                "round": round_number,
                "labeled_pool_size": len(labeled_indices),
                "best_epoch": best_epoch,
                "val_edit": best_val_edit,
                "test_edit": test_edit,
            }
        )
        save_json(run_dir / "history.json", history)

        if not unlabeled_indices:
            print("Entire training set is labeled.")
            break

        queried_indices = query(
            args.query_strategy,
            model,
            unlabeled_indices,
            args.query_batch_size,
            rng,
        )

        save_json(
            round_dir / "queried_samples.json",
            [train_annotations[i]["video"] for i in queried_indices],
        )

        labeled_indices.update(queried_indices)
        unlabeled_indices.difference_update(queried_indices)

        round_number += 1


if __name__ == "__main__":
    main()
