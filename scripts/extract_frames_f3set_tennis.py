"""Extract F3Set-Tennis clips from the original match videos."""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

DEFAULT_DIMENSION = 224
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Extract the frames referenced by the F3Set-Tennis train, "
            "validation, and test annotations."
        )
    )
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        required=True,
        help="Directory containing the original F3Set-Tennis MP4 videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to create one frame directory per clip.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=project_root / "src" / "F3Set" / "data" / "f3set-tennis",
        help="Directory containing train.json, val.json, and test.json.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        help="Output frame height (default: %(default)s).",
    )
    return parser.parse_args()


def clip_details(clip_name: str) -> tuple[str, int, int]:
    """Return the source video stem and half-open frame interval for a clip."""
    try:
        video_stem, start_text, end_text = clip_name.rsplit("_", 2)
        return video_stem, int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid clip name {clip_name!r}; expected '<video>_<start>_<end>'"
        ) from exc


def save_frames(
    capture: cv2.VideoCapture,
    start: int,
    end: int,
    output_dir: Path,
    dimension: int,
) -> None:
    """Save frames in ``[start, end)`` with the same layout as F3Set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    for output_index in range(end - start):
        ok, frame = capture.read()
        if not ok or frame is None:
            source_index = start + output_index
            raise RuntimeError(f"Could not read source frame {source_index}")

        height, width = frame.shape[:2]
        resized_width = width * dimension // height
        resized = cv2.resize(frame, (resized_width, dimension))
        frame_path = output_dir / f"{output_index:06d}.jpg"
        if not cv2.imwrite(str(frame_path), resized):
            raise RuntimeError(f"Could not write frame to {frame_path}")


def extract_dataset(
    input_data_dir: Path,
    output_dir: Path,
    annotations_dir: Path,
    dimension: int,
) -> None:
    if dimension <= 0:
        raise ValueError("--dimension must be greater than zero")
    if not input_data_dir.is_dir():
        raise FileNotFoundError(f"Input data directory not found: {input_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        annotation_path = annotations_dir / f"{split}.json"
        with annotation_path.open() as annotation_file:
            clips = json.load(annotation_file)

        print(split)
        for clip in tqdm(clips, desc=split, unit="clip"):
            clip_name = clip["video"]
            video_stem, start, end = clip_details(clip_name)
            video_path = input_data_dir / f"{video_stem}.mp4"

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open video: {video_path}")

            try:
                save_frames(
                    capture,
                    start,
                    end,
                    output_dir / clip_name,
                    dimension,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to extract clip {clip_name!r} from {video_path}"
                ) from exc
            finally:
                capture.release()


def main() -> None:
    args = parse_args()
    extract_dataset(
        args.input_data_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.annotations_dir.expanduser().resolve(),
        args.dimension,
    )


if __name__ == "__main__":
    main()
