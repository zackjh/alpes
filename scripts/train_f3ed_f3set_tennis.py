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
from src.F3Set.util.eval import non_maximum_suppression_np
from torch.backends import cudnn
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

# Use deterministic settings for reproducibility
# CUDA 10.2+ requires a deterministic cuBLAS workspace configuration when
# torch.use_deterministic_algorithms(True) is enabled.  Set it before the first
# CUDA operation; preserve either supported value if the caller supplied one.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
cudnn.benchmark = False
cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

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

QUERY_SCORE_KEYS = {
    "COARSE_UNCERTAINTY_MEASURE": "coarse_uncertainty_measure",
    "COARSE_ENTROPY_MEASURE": "coarse_entropy_measure",
    "FINE_UNCERTAINTY_MEASURE": "fine_uncertainty_measure",
    "FINE_ENTROPY_MEASURE": "fine_entropy_measure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pool-based active learning for F3ED on F3Set-Tennis"
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
    parser.add_argument(
        "--query_batch_size",
        type=float,
        required=True,
        metavar="PERCENT",
        help="per-query frame budget as a percentage of the training set",
    )
    parser.add_argument(
        "--query_strategy",
        required=True,
        choices=[
            *QUERY_SCORE_KEYS,
            "RANDOM_SAMPLING",
        ],
    )
    parser.add_argument(
        "--query_score_pooling",
        choices=["MAX", "MEAN"],
        default="MAX",
        help="temporal pooling used to turn frame query scores into clip scores",
    )
    return parser.parse_args()


def save_json(path: Path, data: object) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def frame_budget_from_percentage(percentage: float, total_frames: int) -> int:
    """Convert a percentage of the annotation budget to a frame target."""
    return math.ceil(total_frames * percentage / 100)


def select_clips_to_frame_budget(
    ordered_indices: list[int],
    frame_counts: list[int],
    frame_budget: int,
) -> list[int]:
    """Take whole clips in order until their frames meet the target budget."""
    selected = []
    selected_frames = 0
    for index in ordered_indices:
        selected.append(index)
        selected_frames += frame_counts[index]
        if selected_frames >= frame_budget:
            break
    return selected


def random_clips_to_frame_budget(
    candidate_indices: set[int],
    frame_counts: list[int],
    frame_budget: int,
    rng: random.Random,
) -> list[int]:
    """Randomly order candidates and select whole clips to a frame budget."""
    candidates = sorted(candidate_indices)
    rng.shuffle(candidates)
    return select_clips_to_frame_budget(candidates, frame_counts, frame_budget)


def uncertainty_measure(probabilities: np.ndarray) -> np.ndarray:
    """Return the paper's uncertainty measure for binary probabilities."""
    return 1.0 - 2.0 * np.abs(probabilities - 0.5)


def normalized_entropy(probabilities: np.ndarray, axis: int) -> np.ndarray:
    """Return entropy normalized to [0, 1] for binary distributions."""
    entropy_terms = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0
    entropy_terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    return entropy_terms.sum(axis=axis) / math.log(2.0)


def pool_frame_scores(scores: np.ndarray, pooling: str) -> float:
    """Pool a non-empty vector of frame scores into a clip score."""
    if scores.size == 0:
        raise ValueError("cannot pool an empty score vector")
    if pooling == "MAX":
        return float(np.max(scores))
    if pooling == "MEAN":
        return float(np.mean(scores))
    raise ValueError(f"Unknown query score pooling: {pooling}")


def calculate_query_scores(
    coarse_probabilities: np.ndarray,
    fine_probabilities: np.ndarray,
    predicted_event_mask: np.ndarray,
    pooling: str,
) -> dict[str, float]:
    """Calculate the coarse and fine UM/EM clip scores."""
    if coarse_probabilities.ndim != 2 or coarse_probabilities.shape[1] != 2:
        raise ValueError("coarse probabilities must have shape (frames, 2)")
    if fine_probabilities.ndim != 2:
        raise ValueError("fine probabilities must have shape (frames, classes)")
    if fine_probabilities.shape[0] != coarse_probabilities.shape[0]:
        raise ValueError("coarse and fine probabilities must have equal frame counts")
    if predicted_event_mask.shape != (coarse_probabilities.shape[0],):
        raise ValueError("predicted event mask must have shape (frames,)")

    coarse_confidence = np.max(coarse_probabilities, axis=1)
    coarse_um_frames = uncertainty_measure(coarse_confidence)
    coarse_em_frames = normalized_entropy(coarse_probabilities, axis=1)

    # Each fine output is an independent Bernoulli probability. Average the
    # per-attribute measures so that the result remains on a [0, 1] scale.
    fine_um_frames = uncertainty_measure(fine_probabilities).mean(axis=1)
    fine_binary_distributions = np.stack(
        (fine_probabilities, 1.0 - fine_probabilities),
        axis=-1,
    )
    fine_em_frames = normalized_entropy(fine_binary_distributions, axis=2).mean(axis=1)

    coarse_um = pool_frame_scores(coarse_um_frames, pooling)
    coarse_em = pool_frame_scores(coarse_em_frames, pooling)

    if np.any(predicted_event_mask):
        fine_um = pool_frame_scores(fine_um_frames[predicted_event_mask], pooling)
        fine_em = pool_frame_scores(fine_em_frames[predicted_event_mask], pooling)
    else:
        fine_um = 0.0
        fine_em = 0.0

    return {
        "coarse_uncertainty_measure": coarse_um,
        "fine_uncertainty_measure": fine_um,
        "coarse_entropy_measure": coarse_em,
        "fine_entropy_measure": fine_em,
    }


def score_unlabeled_clips(
    model: F3Set,
    dataset: ActionSeqVideoDataset,
    pooling: str,
) -> dict[str, dict[str, float]]:
    """Run F3ED over the unlabeled pool and score every annotation clip."""
    predictions = {
        video: (
            np.zeros((video_len, 2), np.float64),
            np.zeros((video_len, model._num_classes), np.float64),
            np.zeros(video_len, np.int64),
        )
        for video, video_len, _ in dataset.videos
    }

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=BASE_NUM_WORKERS * 2,
    )
    for batch in loader:
        _, batch_coarse_scores, batch_fine_scores = model.predict(
            batch["frame"], batch["hand"]
        )
        for batch_index, video in enumerate(batch["video"]):
            coarse_scores, fine_scores, support = predictions[video]
            clip_coarse_scores = batch_coarse_scores[batch_index]
            clip_fine_scores = batch_fine_scores[batch_index]
            start = int(batch["start"][batch_index])

            if start < 0:
                clip_coarse_scores = clip_coarse_scores[-start:]
                clip_fine_scores = clip_fine_scores[-start:]
                start = 0

            end = min(start + len(clip_coarse_scores), len(coarse_scores))
            valid_length = end - start
            if valid_length <= 0:
                continue
            coarse_scores[start:end] += clip_coarse_scores[:valid_length]
            fine_scores[start:end] += clip_fine_scores[:valid_length]
            support[start:end] += 1

    scores_by_video = {}
    for video, (coarse_scores, fine_scores, support) in predictions.items():
        if np.any(support == 0):
            raise RuntimeError(f"inference did not cover every frame of {video}")
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]
        predicted_event_mask = np.argmax(
            non_maximum_suppression_np(coarse_scores.copy(), WINDOW),
            axis=1,
        ).astype(bool)
        scores_by_video[video] = calculate_query_scores(
            coarse_scores,
            fine_scores,
            predicted_event_mask,
            pooling,
        )
    return scores_by_video


def select_scored_clips_to_frame_budget(
    candidate_indices: set[int],
    scores: dict[int, dict[str, float]],
    score_key: str,
    frame_counts: list[int],
    frame_budget: int,
) -> list[int]:
    """Rank by descending score and select whole clips to the frame budget."""
    ordered_indices = sorted(
        candidate_indices,
        key=lambda index: (-scores[index][score_key], index),
    )
    return select_clips_to_frame_budget(ordered_indices, frame_counts, frame_budget)


def query(
    strategy: str,
    model: F3Set,
    unlabeled_indices: set[int],
    query_frame_budget: int,
    frame_counts: list[int],
    rng: random.Random,
    unlabeled_data: ActionSeqVideoDataset | None = None,
    index_by_video: dict[str, int] | None = None,
    score_pooling: str = "MAX",
) -> tuple[list[int], dict[int, dict[str, float]] | None]:
    """Select samples from the unlabeled pool."""
    if strategy == "RANDOM_SAMPLING":
        return (
            random_clips_to_frame_budget(
                unlabeled_indices,
                frame_counts,
                query_frame_budget,
                rng,
            ),
            None,
        )

    if unlabeled_data is None or index_by_video is None:
        raise ValueError("active query strategies require an unlabeled dataset")

    score_key = QUERY_SCORE_KEYS.get(strategy)
    if score_key is None:
        raise ValueError(f"Unknown query strategy: {strategy}")

    scores_by_video = score_unlabeled_clips(model, unlabeled_data, score_pooling)
    scores = {
        index_by_video[video]: values for video, values in scores_by_video.items()
    }
    if set(scores) != unlabeled_indices:
        raise RuntimeError("scored pool does not match the unlabeled pool")

    selected = select_scored_clips_to_frame_budget(
        unlabeled_indices,
        scores,
        score_key,
        frame_counts,
        query_frame_budget,
    )
    return selected, scores


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
        "rny002_tsm",
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

    if not 0 < args.initial_labeled_pool_size <= 100:
        raise ValueError("initial labeled pool size must be in (0, 100] percent")

    if not 0 < args.query_batch_size <= 100:
        raise ValueError("query batch size must be in (0, 100] percent")

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

    frame_counts = [annotation["num_frames"] for annotation in train_annotations]
    total_training_frames = sum(frame_counts)
    initial_frame_budget = frame_budget_from_percentage(
        args.initial_labeled_pool_size,
        total_training_frames,
    )
    query_frame_budget = frame_budget_from_percentage(
        args.query_batch_size,
        total_training_frames,
    )

    if initial_frame_budget > total_training_frames:
        raise ValueError(
            "initial labeled pool frame budget cannot exceed the training pool"
        )

    # Seed the experiment while otherwise leaving F3ED's training behaviour
    # unchanged.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rng = random.Random(args.seed)

    all_indices = set(range(len(train_annotations)))
    labeled_indices = set(
        random_clips_to_frame_budget(
            all_indices,
            frame_counts,
            initial_frame_budget,
            rng,
        )
    )
    unlabeled_indices = all_indices - labeled_indices

    classes = load_classes(str(dataset_root / "elements.txt"))

    save_json(
        run_dir / "config.json",
        {
            "name": args.name,
            "seed": args.seed,
            "query_strategy": args.query_strategy,
            "query_score_pooling": args.query_score_pooling,
            "initial_labeled_pool_size": args.initial_labeled_pool_size,
            "query_batch_size": args.query_batch_size,
            "active_learning_budget_unit": "percent_of_training_frames",
            "total_training_frames": total_training_frames,
            "initial_labeled_pool_frame_budget": initial_frame_budget,
            "query_batch_frame_budget": query_frame_budget,
            "f3ed": {
                "feature_arch": "rny002_tsm",
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

        labeled_frames = sum(frame_counts[i] for i in labeled_indices)
        unlabeled_frames = sum(frame_counts[i] for i in unlabeled_indices)
        print(
            f"\n=== Round {round_number} ===\n"
            f"Labeled:   {len(labeled_indices)} clips / {labeled_frames} frames\n"
            f"Unlabeled: {len(unlabeled_indices)} clips / {unlabeled_frames} frames"
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
                "labeled_pool_frames": labeled_frames,
                "labeled_budget_percent": (
                    100 * labeled_frames / total_training_frames
                ),
                "best_epoch": best_epoch,
                "val_edit": best_val_edit,
                "test_edit": test_edit,
            }
        )
        save_json(run_dir / "history.json", history)

        if not unlabeled_indices:
            print("Entire training set is labeled.")
            break

        unlabeled_data = None
        index_by_video = None
        if args.query_strategy != "RANDOM_SAMPLING":
            unlabeled_file = round_dir / "unlabeled_pool.json"
            unlabeled_annotations = [
                train_annotations[i] for i in sorted(unlabeled_indices)
            ]
            save_json(unlabeled_file, unlabeled_annotations)
            index_by_video = {
                annotation["video"]: index
                for index, annotation in enumerate(train_annotations)
                if index in unlabeled_indices
            }
            if len(index_by_video) != len(unlabeled_indices):
                raise ValueError("training video names must be unique")
            unlabeled_data = ActionSeqVideoDataset(
                classes,
                str(unlabeled_file),
                str(frame_dir),
                CLIP_LEN,
                crop_dim=CROP_DIM,
                stride=STRIDE,
                overlap_len=0,
            )

        queried_indices, query_scores = query(
            args.query_strategy,
            model,
            unlabeled_indices,
            query_frame_budget,
            frame_counts,
            rng,
            unlabeled_data=unlabeled_data,
            index_by_video=index_by_video,
            score_pooling=args.query_score_pooling,
        )

        if query_scores is not None:
            selected_indices = set(queried_indices)
            save_json(
                round_dir / "query_scores.json",
                [
                    {
                        "index": index,
                        "video": train_annotations[index]["video"],
                        "selected": index in selected_indices,
                        **query_scores[index],
                    }
                    for index in sorted(query_scores)
                ],
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
