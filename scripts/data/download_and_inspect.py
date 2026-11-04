import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "tm21cy/NYT-Connections"
FILENAME = "ConnectionsFinalDataset.json"
LOCAL_DATA_DIR = Path("data/raw")


def main() -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading raw dataset file...")

    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )

    print(f"Cached file: {cached_path}")

    local_path = LOCAL_DATA_DIR / FILENAME
    local_path.write_bytes(Path(cached_path).read_bytes())

    print(f"Copied dataset to: {local_path.resolve()}")

    with local_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    print("\nDataset loaded successfully.")
    print(f"Top-level Python type: {type(records).__name__}")
    print(f"Number of records: {len(records)}")

    if not records:
        raise RuntimeError("Dataset is empty.")

    print("\nAvailable fields:")
    print(list(records[0].keys()))

    print("\nFirst record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))

    print("\nSecond record:")
    print(json.dumps(records[1], indent=2, ensure_ascii=False))

    print("\nDate examples:")
    for record in records[:10]:
        print(repr(record.get("date")))


if __name__ == "__main__":
    main()
