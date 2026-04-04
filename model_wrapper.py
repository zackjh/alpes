import json
from pathlib import Path
import subprocess


class F3EDTennis:
    def __init__(
        self,
        f3set_repo_root: Path,
    ):
        self.f3set_repo_root = f3set_repo_root

    def train(
        self,
        results_path: Path,
        num_of_epochs: int,
        start_val_epoch: int,
        train_json_path: Path,
    ):
        cmd = [
            "python3",
            "train_f3set_f3ed.py",
            "f3set-tennis",
            str(self.f3set_repo_root / "data" / "f3set-tennis" / "extracted_frames"),
            "-s",
            str(results_path),
            "-m",
            "rny002_tsm",
            "-t",
            "gru",
            "--num_epochs",
            str(num_of_epochs),
            "--start_val_epoch",
            str(start_val_epoch),
            "--train_json_path",
            str(train_json_path),
            "--val_json_path",
            str(self.f3set_repo_root / "data" / "f3set-tennis" / "val.json"),
            "--skip_test_eval",
        ]

        print("[LOG][model.py] Running training script:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.f3set_repo_root, check=True)

    def test(self, trained_model_path: Path):
        cmd = [
            "python3",
            "test_f3set_f3ed.py",
            str(trained_model_path),
            str(self.f3set_repo_root / "data" / "f3set-tennis" / "extracted_frames"),
            "-s",
            "test",
        ]

        print("[LOG][model.py] Running test script:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.f3set_repo_root, check=True)

    def get_query_batch_video_names(
        self,
        trained_model_path: Path,
        unlabeled_data_json_path: Path,
        query_batch_size: int,
    ) -> list[str]:
        cmd = [
            "python3",
            "get_scores.py",
            str(trained_model_path),
            str(unlabeled_data_json_path),
        ]

        print("[LOG][model.py] Running get_query_batch script:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.f3set_repo_root, check=True)

        max_uncertainty_per_video_json_path = (
            trained_model_path / "max_uncertainty_per_video.json"
        )
        with open(max_uncertainty_per_video_json_path, "r", encoding="utf-8") as f:
            max_uncertainty_scores = json.load(f)

        effective_query_batch_size = min(len(max_uncertainty_scores), query_batch_size)

        max_uncertainty_scores_sorted = sorted(
            max_uncertainty_scores.items(), key=lambda x: x[1], reverse=True
        )

        # This is the query batch - it's the top k videos with the highest uncertainty scores
        top_k = max_uncertainty_scores_sorted[:effective_query_batch_size]
        query_batch_video_names = [x[0] for x in top_k]

        # Save top k samples as into `query_batch.json` for logging
        with open(trained_model_path / "query_batch.json", "w", encoding="utf-8") as f:
            json.dump(top_k, f, indent=2)

        # Return only the video names
        return query_batch_video_names
