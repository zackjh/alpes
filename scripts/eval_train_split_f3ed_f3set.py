"""Score the F3Set-Tennis training split with an F3ED checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from src.F3Set.dataset.frame_process import ActionSeqVideoDataset
from src.F3Set.train_f3set_f3ed import F3Set
from src.F3Set.util.dataset import load_classes

from train_f3ed_f3set_tennis import (
    CLIP_LEN,
    CROP_DIM,
    STRIDE,
    WINDOW,
    score_unlabeled_clips_for_poolings,
)

project_root = Path(__file__).resolve().parents[1]
DATASET_ROOT = project_root / "src" / "F3Set" / "data" / "f3set-tennis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an F3ED checkpoint on the F3Set-Tennis training split "
            "and write the active-learning query scores"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to an F3ED checkpoint.pt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path at which to write the query-score JSON file",
    )
    parser.add_argument(
        "--query-score-pooling",
        "--query_score_pooling",
        nargs="+",
        choices=["MAX", "MEAN"],
        default=["MEAN", "MAX"],
        help=(
            "temporal pooling methods to record for each video "
            "(default: MEAN MAX)"
        ),
    )
    parser.add_argument(
        "--frame_dir",
        type=Path,
        required=True,
        help="Directory containing the F3Set-Tennis per-clip frame folders.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict]:
    with path.open() as file:
        annotations = json.load(file)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected a list of annotations in {path}")
    return annotations


def save_json(path: Path, value: object) -> None:
    """Write JSON atomically so an interrupted write cannot corrupt the output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    frame_dir = args.frame_dir.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    train_file = DATASET_ROOT / "train.json"
    annotations = load_annotations(train_file)
    videos = [annotation["video"] for annotation in annotations]
    if len(set(videos)) != len(videos):
        raise ValueError("Training video names must be unique")

    classes = load_classes(str(DATASET_ROOT / "elements.txt"))
    train_data = ActionSeqVideoDataset(
        classes,
        str(train_file),
        str(frame_dir),
        CLIP_LEN,
        crop_dim=CROP_DIM,
        stride=STRIDE,
        # Match the active-learning pool inference configuration.
        overlap_len=0,
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
    model.load(torch.load(checkpoint, map_location="cuda"))

    print(
        f"Scoring {len(annotations)} training videos with "
        f"{' and '.join(args.query_score_pooling)} pooling...",
        flush=True,
    )
    scores_by_video = score_unlabeled_clips_for_poolings(
        model,
        train_data,
        tuple(args.query_score_pooling),
    )
    if set(scores_by_video) != set(videos):
        raise RuntimeError("Scored videos do not match the F3Set-Tennis training split")

    results = [
        {
            "index": index,
            "video": video,
            **{
                pooling.lower(): scores_by_video[video][pooling]
                for pooling in args.query_score_pooling
            },
        }
        for index, video in enumerate(videos)
    ]
    save_json(output, results)
    print(f"Wrote query scores to {output}")


if __name__ == "__main__":
    main()
