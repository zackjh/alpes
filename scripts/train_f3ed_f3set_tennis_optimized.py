"""Faster F3ED active-learning training on F3Set-Tennis.

This keeps the original optimizer updates, data sampling, augmentations, learning
rate schedule, and validation-based checkpoint selection.  It omits the
per-epoch validation-loss pass (which never affects training or checkpoint
selection) and test-set evaluation.  Checkpoints use the same state-dict format
as train_f3ed_f3set_tennis.py.
"""

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.F3Set.dataset.frame_process import (
    ActionSeqDataset,
    ActionSeqVideoDataset,
)
from src.F3Set.model.common import step
from src.F3Set.train_f3set_f3ed import F3Set, evaluate
from src.F3Set.util.dataset import load_classes
from torch.backends import cudnn
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_f3ed_f3set_tennis import (
    BASE_NUM_WORKERS,
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
    parse_args,
    query,
    random_clips_to_frame_budget,
    save_json,
)

# Optimize for speed at the expense of reproducibility
cudnn.benchmark = True
cudnn.deterministic = False
torch.use_deterministic_algorithms(False)


def _without_nan(value: torch.Tensor) -> torch.Tensor:
    """Match the original scalar NaN check without synchronizing the GPU."""
    return torch.where(torch.isnan(value), torch.zeros_like(value), value)


class OptimizedF3Set(F3Set):
    """F3Set with synchronization-free but equivalent hot paths."""

    class Impl(F3Set.Impl):
        def forward(
            self,
            frame,
            coarse_label=None,
            fine_label=None,
            hand=None,
            max_seq_len=20,
        ):
            batch_size, true_clip_len, channels, height, width = frame.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                assert true_clip_len <= self._require_clip_len, (
                    f"Expected {self._require_clip_len}, got {true_clip_len}"
                )
                if true_clip_len < self._require_clip_len:
                    frame = F.pad(
                        frame,
                        (0,) * 7 + (self._require_clip_len - true_clip_len,),
                    )
                    clip_len = self._require_clip_len

            if self._is_3d:
                im_feat = self._glb_feat(frame.transpose(1, 2)).transpose(1, 2)
            else:
                im_feat = self._glb_feat(
                    frame.view(-1, channels, height, width)
                ).reshape(batch_size, clip_len, -1)

            enc_feat = self._head(im_feat)
            coarse_pred = self._coarse_pred(enc_feat)
            fine_pred = self._fine_pred(enc_feat)

            if not self._use_ctx:
                return coarse_pred, fine_pred, None, None, None

            coarse_pred_score = torch.softmax(coarse_pred, dim=2)
            fine_pred_score = torch.sigmoid(fine_pred).to(dtype=fine_pred.dtype)

            if coarse_label is None:
                from src.F3Set.util.eval import non_maximum_suppression

                coarse_label = non_maximum_suppression(coarse_pred_score, self._window)
                coarse_label = torch.argmax(coarse_label, dim=2)
            else:
                coarse_label = coarse_pred_score * coarse_label.unsqueeze(-1)
                coarse_label = torch.argmax(coarse_label, dim=2)

            if fine_label is None:
                fine_label = fine_pred_score
            if hand is None:
                raise ValueError("hand is required when the context module is enabled")

            seq_pred = torch.zeros(
                batch_size,
                max_seq_len,
                self._num_classes + 1,
                dtype=fine_label.dtype,
                device=self._device,
            )
            seq_label = torch.zeros_like(seq_pred)
            seq_mask = torch.ones(
                (batch_size, max_seq_len),
                dtype=torch.bool,
                device=self._device,
            )

            for i in range(batch_size):
                event_mask = coarse_label[i].bool()
                selected_label = fine_label[i, event_mask]
                selected_pred = fine_pred_score[i, event_mask]
                event_count = selected_label.shape[0]

                seq_label[i, :event_count, 1:] = selected_label
                seq_pred[i, :event_count, 1:] = selected_pred

                # The original implementation converts each CUDA index to a
                # Python int.  Batched indexing produces the same values while
                # avoiding a GPU synchronization for every detected event.
                pred_hand_index = torch.round(selected_pred[:, 0]).long()
                label_hand_index = torch.round(selected_label[:, 0]).long()
                seq_pred[i, :event_count, 0] = hand[i].gather(0, pred_hand_index)
                seq_label[i, :event_count, 0] = hand[i].gather(0, label_hand_index)
                seq_mask[i, :event_count] = False

            seq_pred_refined = self._ctx(seq_pred)
            return coarse_pred, fine_pred, seq_pred_refined, seq_label, seq_mask

    def __init__(
        self,
        num_classes,
        feature_arch,
        temporal_arch,
        clip_len,
        step=1,
        window=5,
        use_ctx=True,
        device="cuda",
        multi_gpu=False,
    ):
        self._device = device
        self._multi_gpu = multi_gpu
        self._window = window
        self._use_ctx = use_ctx
        self._model = OptimizedF3Set.Impl(
            num_classes,
            feature_arch,
            temporal_arch,
            clip_len,
            step=step,
            window=window,
            use_ctx=use_ctx,
            device=device,
        )

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    @staticmethod
    def _load_frame_gpu(dataset, batch, device):
        # DataLoader pinning makes these copies asynchronous.  Operations on
        # the default CUDA stream still observe the completed copy.
        frame = batch["frame"].to(device, non_blocking=True)
        if dataset._gpu_transform is not None:
            with torch.no_grad():
                for i in range(frame.shape[0]):
                    frame[i] = dataset._gpu_transform(frame[i])
        return frame

    def epoch(
        self,
        loader,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
        acc_grad_iter=1,
        fg_weight=5,
        epoch=0,
    ):
        del epoch  # Sampling is controlled by the DataLoader workers.

        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs["weight"] = torch.tensor(
                [1, fg_weight], dtype=torch.float32, device=self._device
            )

        # Float64 matches Python's float accumulation while allowing a single
        # device synchronization at the end of the epoch.
        epoch_loss = torch.zeros((), dtype=torch.float64, device=self._device)

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = self._load_frame_gpu(loader.dataset, batch, self._device)
                coarse_label = batch["coarse_label"].to(self._device, non_blocking=True)
                fine_label = batch["fine_label"].to(
                    self._device, dtype=torch.float32, non_blocking=True
                )
                hand = batch["hand"].to(
                    self._device, dtype=torch.float32, non_blocking=True
                )

                with torch.autocast(device_type="cuda"):
                    coarse_pred, fine_pred, seq_pred, seq_label, seq_mask = self._model(
                        frame,
                        coarse_label,
                        fine_label,
                        hand=hand,
                    )

                    coarse_loss = F.cross_entropy(
                        coarse_pred.reshape(-1, 2),
                        coarse_label.flatten(),
                        **ce_kwargs,
                    )
                    fine_bce_loss = F.binary_cross_entropy_with_logits(
                        fine_pred,
                        fine_label,
                        reduction="none",
                    )
                    fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                    fine_loss = (fine_bce_loss * fine_mask).sum() / fine_mask.sum()

                    loss = _without_nan(coarse_loss)
                    loss = loss + _without_nan(fine_loss)

                    if self._use_ctx:
                        ctx_loss = F.binary_cross_entropy_with_logits(
                            seq_pred[~seq_mask], seq_label[~seq_mask]
                        )
                        loss = loss + _without_nan(ctx_loss)

                if optimizer is not None:
                    step(
                        optimizer,
                        scaler,
                        loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                    )

                epoch_loss += loss.detach().double()

        return epoch_loss.item() / len(loader)


def _consume_validation_loader_seed() -> None:
    """Preserve the CPU RNG advance caused by constructing a val iterator."""
    torch.empty((), dtype=torch.int64).random_().item()


def train_round(
    classes: dict,
    train_file: Path,
    val_file: Path,
    frame_dir: Path,
    round_dir: Path,
) -> tuple[OptimizedF3Set, int, float]:
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

    model = OptimizedF3Set(
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

        # The original validation-loss DataLoader consumes one value from the
        # main CPU generator per epoch.  Preserve that RNG advance without the
        # expensive pass, so subsequent training augmentation stays aligned.
        _consume_validation_loader_seed()

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
                "val": None,
                "val_edit": val_edit,
            }
        )
        save_json(round_dir / "loss.json", losses)

        print(f"[Epoch {epoch}] Train loss: {train_loss:.5f} Val edit: {val_edit:.5f}")

    assert best_epoch is not None
    model.load(torch.load(checkpoint_file))
    return model, best_epoch, best_val_edit


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
            "initial_labeled_pool_size": args.initial_labeled_pool_size,
            "query_batch_size": args.query_batch_size,
            "active_learning_budget_unit": "percent_of_training_frames",
            "total_training_frames": total_training_frames,
            "initial_labeled_pool_frame_budget": initial_frame_budget,
            "query_batch_frame_budget": query_frame_budget,
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
            "validation_loss_computed": False,
            "test_evaluation_during_training": False,
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

        labeled_file = round_dir / "labeled_train.json"
        save_json(
            labeled_file,
            [train_annotations[i] for i in sorted(labeled_indices)],
        )

        model, best_epoch, best_val_edit = train_round(
            classes,
            labeled_file,
            val_file,
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
            query_frame_budget,
            frame_counts,
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
