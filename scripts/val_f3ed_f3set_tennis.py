"""Evaluate an F3ED checkpoint on F3Set-Tennis validation at G_high.

Usage:
    python scripts/val_f3ed_f3set_tennis.py --checkpoint path/to/checkpoint.pt \
        --frame_dir path/to/frames

Writes <checkpoint_stem>_val_g_high.json next to the checkpoint. Metrics use
the upstream evaluator's exact-frame matching (delta=0) and native scales:
F1 scores in [0, 1] and edit score in [0, 100].
"""

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
        description="Evaluate F3ED on F3Set-Tennis validation at G_high"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--frame_dir",
        type=Path,
        required=True,
        help="Directory containing the F3Set-Tennis per-clip frame folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    frame_dir = args.frame_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"Frame directory does not exist: {frame_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "src" / "F3Set" / "data" / "f3set-tennis"
    classes = load_classes(str(dataset_root / "elements.txt"))
    if len(classes) != 29:
        raise ValueError("G_high evaluation requires all 29 F3Set-Tennis elements")

    val_data = ActionSeqVideoDataset(
        classes,
        str(dataset_root / "val.json"),
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

    # Keeping all element columns gives G_high event and element metrics.
    f1_event, f1_element, edit = evaluate(
        model, val_data, classes, delta=0, window=WINDOW, device="cuda"
    )
    result = {
        "checkpoint": str(checkpoint),
        "frame_dir": str(frame_dir),
        "split": "val",
        "granularity": "G_high",
        "delta": 0,
        "val_f1_event": float(f1_event),
        "val_f1_element": float(f1_element),
        "val_edit": float(edit),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output = checkpoint.with_name(f"{checkpoint.stem}_val_g_high.json")
    temporary_output = output.with_suffix(".json.tmp")
    temporary_output.write_text(serialized)
    temporary_output.replace(output)
    print(serialized, end="")
    print(f"Wrote validation metrics to {output}")


if __name__ == "__main__":
    main()
