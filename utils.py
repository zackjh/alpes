import json
from pathlib import Path
from datapool import DataPool


def import_data_from_json(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def export_data_to_json(export_path: Path, data_pool: DataPool):
    with export_path.open("w", encoding="utf-8") as f:
        json.dump(data_pool.data, f, indent=2)
