"""Evaluate an F3ED F3Set-Tennis checkpoint on the test split."""

import argparse
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from src.F3Set.dataset.frame_process import ActionSeqVideoDataset
from src.F3Set.train_f3set_f3ed import (
    BASE_NUM_WORKERS,
    INFERENCE_BATCH_SIZE,
    F3Set,
)
from src.F3Set.util.dataset import load_classes
from src.F3Set.util.eval import edit_score, non_maximum_suppression_np
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_f3ed_f3set_tennis import CLIP_LEN, CROP_DIM, STRIDE, WINDOW

# F3Set paper, Section 3.3: https://arxiv.org/html/2504.08222v1#S3.SS3
# Column order follows data/f3set-tennis/elements.txt, not the paper's numbering.
GRANULARITY_COLUMNS = {
    "G_high": list(range(29)),
    "G_mid": list(range(24)),
    "G_low": [0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28],
}


def mean_f1(counts):
    """Match the upstream evaluator's macro F1 calculation."""
    counts = np.asarray(counts).reshape(-1, 3)
    if not len(counts):
        return 0.0
    precision = counts[:, 0] / (counts[:, 0] + counts[:, 1] + 1e-10)
    recall = counts[:, 0] / (counts[:, 0] + counts[:, 2] + 1e-10)
    return float(np.mean(2 * precision * recall / (precision + recall + 1e-10)))


def update_metrics(pred, label, element_counts, event_counts, edits):
    """Accumulate the upstream evaluator's exact-frame (delta=0) metrics."""
    element_counts[:, 0] += np.sum((pred == 1) & (label == 1), axis=0)
    element_counts[:, 1] += np.sum((pred == 1) & (label == 0), axis=0)
    element_counts[:, 2] += np.sum((pred == 0) & (label == 1), axis=0)
    weights = 1 << np.arange(pred.shape[1] - 1, -1, -1)
    preds = pred @ weights
    labels = label @ weights
    for predicted, target in zip(preds, labels, strict=True):
        if predicted > 0 and predicted == target:
            event_counts.setdefault(int(predicted), [0, 0, 0])[0] += 1
        # Preserve upstream event-F1 semantics: FP only on background frames.
        if predicted > 0 and target == 0:
            event_counts.setdefault(int(predicted), [0, 0, 0])[1] += 1
        if target > 0 and target != predicted:
            event_counts.setdefault(int(target), [0, 0, 0])[2] += 1
    pred_sequence = [key for key, _ in groupby(preds) if key != 0]
    label_sequence = [key for key, _ in groupby(labels) if key != 0]
    edits.append(edit_score(pred_sequence, label_sequence))


def evaluate_granularities(model, dataset, classes):
    """Run inference once and project decoded events to each paper granularity."""
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, 2), np.float32),
            np.zeros((video_len, len(classes)), np.float32),
            np.zeros(video_len, np.int32),
        )

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    for clip in tqdm(
        DataLoader(
            dataset,
            num_workers=BASE_NUM_WORKERS * 2,
            pin_memory=True,
            batch_size=batch_size,
        )
    ):
        _, batch_coarse_scores, batch_fine_scores = model.predict(
            clip["frame"], clip["hand"]
        )
        for i in range(clip["frame"].shape[0]):
            video = clip["video"][i]
            coarse_scores, fine_scores, support = pred_dict[video]
            coarse_pred_scores = batch_coarse_scores[i]
            fine_pred_scores = batch_fine_scores[i]

            start = clip["start"][i].item()
            if start < 0:
                coarse_pred_scores = coarse_pred_scores[-start:, :]
                fine_pred_scores = fine_pred_scores[-start:, :]
                start = 0
            end = start + coarse_pred_scores.shape[0]
            if end >= coarse_scores.shape[0]:
                end = coarse_scores.shape[0]
                coarse_pred_scores = coarse_pred_scores[: end - start, :]
                fine_pred_scores = fine_pred_scores[: end - start, :]
            coarse_scores[start:end, :] += coarse_pred_scores
            fine_scores[start:end, :] += fine_pred_scores
            support[start:end] += 1

    metrics = {
        name: (np.zeros((len(columns), 3), dtype=int), {}, [])
        for name, columns in GRANULARITY_COLUMNS.items()
    }
    # Event presence is shared by all granularities. Pool TP/FP/FN over videos
    # and match at exact sampled frames (delta=0), like the existing metrics.
    detection_counts = np.zeros(3, dtype=np.int64)
    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        if np.any(support == 0):
            raise RuntimeError(f"Some frames in {video} have no predictions")
        coarse_label, fine_label = dataset.get_labels(video)
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]
        coarse_pred = np.argmax(non_maximum_suppression_np(coarse_scores, 5), axis=1)
        detected = coarse_pred == 1
        actual = coarse_label == 1
        detection_counts += (
            np.count_nonzero(detected & actual),
            np.count_nonzero(detected & ~actual),
            np.count_nonzero(~detected & actual),
        )

        # Decode the full event before selecting a granularity, as in F3ED.evaluate.
        fine_pred = np.zeros_like(fine_scores, dtype=int)
        rows = np.arange(len(fine_scores))
        for start, end in [(0, 2), (2, 5), (5, 8), (16, 24), (25, 29)]:
            fine_pred[rows, start + np.argmax(fine_scores[:, start:end], axis=1)] = 1
        fine_pred[:, 24] = fine_scores[:, 24] > 0.5
        non_serve = rows[fine_pred[:, 5] != 1]
        for start, end in [(8, 10), (10, 16)]:
            fine_pred[
                non_serve,
                start + np.argmax(fine_scores[non_serve, start:end], axis=1),
            ] = 1
        fine_pred *= coarse_pred[:, None]

        for name, columns in GRANULARITY_COLUMNS.items():
            update_metrics(
                fine_pred[:, columns], fine_label[:, columns], *metrics[name]
            )

    tp, fp, fn = detection_counts
    denominator = 2 * tp + fp + fn
    class_agnostic_f1 = float(2 * tp / denominator) if denominator else 0.0
    return {
        name: {
            "test_f1_event": mean_f1(list(event_counts.values())),
            "test_f1_element": mean_f1(element_counts),
            "test_f1_class_agnostic": class_agnostic_f1,
            "test_edit": float(np.mean(edits)) if edits else 0.0,
        }
        for name, (element_counts, event_counts, edits) in metrics.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an F3ED checkpoint on F3Set-Tennis test data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to write all three granularities as JSON",
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

    metrics = evaluate_granularities(model, test_data, classes)
    result = {
        "checkpoint": str(checkpoint),
        # Preserve existing top-level metrics as aliases for G_high.
        **metrics["G_high"],
        **metrics,
    }
    print(json.dumps(result, indent=2))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
