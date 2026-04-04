"""
Performs inference to obtain coarse and fine scores.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.frame_process import ActionSeqVideoDataset
from util.io import load_json
from util.dataset import load_classes
from train_f3set_f3ed import F3Set

# CONSTANTS - DON'T TOUCH FOR NOW
BASE_NUM_WORKERS = 4
INFERENCE_BATCH_SIZE = 4

config = {
    "batch_size": 4,
    "clip_len": 96,
    "crop_dim": 224,
    "dataset": "f3set-tennis",
    # "epoch_num_frames": 500000,
    "feature_arch": "rny002_tsm",
    "gpu_parallel": False,
    "learning_rate": 0.001,
    "num_classes": 29,
    # "num_epochs": 25,
    # "start_val_epoch": 15,
    "stride": 2,
    "temporal_arch": "gru",
    "test_json_path": None,
    "train_json_path": "THIS SHOULDN'T MATTER",
    "use_ctx": False,
    "val_json_path": "THIS SHOULDN'T MATTER",
    "warm_up_epochs": 3,
    "window": 5,
}

my_classes = load_classes(
    Path("/mnt/ssd2/zachary/alpes/f3set/data/f3set-tennis/elements.txt").resolve()
)

frame_dir = Path(
    "/mnt/ssd2/zachary/alpes/f3set/data/f3set-tennis/extracted_frames"
).resolve()

# VARIABLES
best_epoch_checkpoint_path = Path(
    "/mnt/ssd2/zachary/alpes/experiments/random_sampling_1/active_learning_iteration_0/checkpoint_024.pt"
).resolve()


# HELPER FUNCTIONS
def get_best_epoch(model_dir, key="val_edit"):
    data = load_json(os.path.join(model_dir, "loss.json"))
    best = max(data, key=lambda x: x[key])
    return best["epoch"]


def get_coarse_and_fine_scores_per_video(
    model_directory: Path, split_json_path: Path, classes
):
    """
    Get the raw coarse and fine scores for each video in the given datset.

    These coarse and fine scores will be used to create the active learning query strategy.
    """
    # Preprocessing
    model = F3Set(
        len(my_classes),
        config["feature_arch"],
        config["temporal_arch"],
        clip_len=config["clip_len"],
        step=config["stride"],
        window=config["window"],
        use_ctx=config["use_ctx"],
        multi_gpu=config["gpu_parallel"],
    )

    best_epoch = get_best_epoch(model_directory)

    best_epoch_checkpoint_path = model_directory / "checkpoint_{:03d}.pt".format(
        best_epoch
    )

    model.load(torch.load(best_epoch_checkpoint_path))

    dataset = ActionSeqVideoDataset(
        my_classes,
        split_json_path,
        frame_dir,
        config["clip_len"],
        overlap_len=config["clip_len"] // 2,
        crop_dim=config["crop_dim"],
        stride=config["stride"],
    )

    pred_dict = {}

    for video, video_len, _ in dataset.videos:
        # dataset.videos is a list of tuples, with each tuple being of size 3
        # Each tuple has the format: (video_name: str, video_len_after_stride: int, fps_after_stride: float)
        # E.g.
        # [('20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_100393_100484', 45, 12.5),
        # ('20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_102829_102930', 50, 12.5), ...]

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

        if batch_size > 1:
            # Batched by dataloader
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

    coarse_and_fine_scores_per_video = {}

    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        # Normalise the scores
        # Sliding window causes the same frame to be processed multiple times
        # This is where we undo that effect
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        coarse_and_fine_scores_per_video[video] = {
            "coarse_scores": coarse_scores,
            "fine_scores": fine_scores,
        }

    return coarse_and_fine_scores_per_video


def export_query_batch_by_uncertainty_as_json(
    model_directory: Path, coarse_and_fine_scores
):
    max_uncertainty_per_video = {}
    for video_name, inner_dict in coarse_and_fine_scores.items():
        coarse_scores = inner_dict["coarse_scores"]

        uncertainty_scores = 1 - np.max(coarse_scores, axis=1)
        max_uncertainty = np.max(uncertainty_scores)

        # Add to result dictionary
        max_uncertainty_per_video[video_name] = float(max_uncertainty)

    # Export as a JSON file since we can't directly pass this to the active learning experiment
    output_json_path = model_directory / "max_uncertainty_per_video.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(max_uncertainty_per_video, f, indent=2)
    print(f"[LOG][get_scores.py] {output_json_path} was created.")


if __name__ == "__main__":

    model_directory = Path(sys.argv[1]).resolve()
    unlabeled_data_json_path = Path(sys.argv[2]).resolve()

    coarse_and_fine_scores_per_video = get_coarse_and_fine_scores_per_video(
        model_directory,
        unlabeled_data_json_path,
        my_classes,
    )

    export_query_batch_by_uncertainty_as_json(
        model_directory, coarse_and_fine_scores_per_video
    )
