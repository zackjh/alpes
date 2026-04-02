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
