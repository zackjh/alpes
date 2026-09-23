"""Evaluate an F3ED F3Set-Tennis checkpoint on the test split."""

import argparse
import json
from pathlib import Path

import torch
from src.F3Set.dataset.frame_process import ActionSeqVideoDataset
from src.F3Set.train_f3set_f3ed import F3Set, evaluate
from src.F3Set.util.dataset import load_classes

from train_f3ed_f3set_tennis import CLIP_LEN, CROP_DIM, STRIDE, WINDOW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an F3ED checkpoint on F3Set-Tennis test data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to write the test metric as JSON",
    )
    parser.add_argument(
        "--frame_dir",
        type=Path,
        required=True,
        help="Directory containing the F3Set-Tennis per-clip frame folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "src" / "F3Set" / "data" / "f3set-tennis"
    frame_dir = args.frame_dir.expanduser().resolve()
    test_file = dataset_root / "test.json"
    classes = load_classes(str(dataset_root / "elements.txt"))

    test_data = ActionSeqVideoDataset(
        classes,
        str(test_file),
        str(frame_dir),
        CLIP_LEN,
        crop_dim=CROP_DIM,
        stride=STRIDE,
        overlap_len=CLIP_LEN // 2,
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

    test_f1_event, test_f1_element, test_edit = evaluate(
        model,
        test_data,
        classes,
        window=WINDOW,
    )
    result = {
        "checkpoint": str(checkpoint),
        "test_f1_event": test_f1_event,
        "test_f1_element": test_f1_element,
        "test_edit": test_edit,
    }
    print(json.dumps(result, indent=2))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
