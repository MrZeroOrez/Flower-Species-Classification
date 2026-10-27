"""Download the user-selected Kaggle dataset and create a deduplicated split.

Run: python -m scripts.prepare_data
No dataset files are committed to Git. Review source terms before redistribution.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

SOURCE = "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/alxmamaev/flowers-recognition"
CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root, seed=42):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "flowers-recognition.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urlopen(DOWNLOAD, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    raw = Counter()
    with ZipFile(archive) as bundle:
        if sum(i.file_size for i in bundle.infolist()) > 1024 ** 3:
            raise ValueError("Archive exceeds the 1 GiB extraction limit.")
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (len(path.parts) != 3 or path.parts[0] != "flowers"
                    or path.parts[1] not in CLASSES or path.name in {".", ".."}
                    or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}):
                raise ValueError(f"Unexpected archive entry: {entry.filename}")
            destination = root.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            raw[path.parts[1]] += 1

    groups = defaultdict(list)
    invalid = []
    # Hash decoded pixels: byte-different copies of identical images stay together.
    for label in CLASSES:
        for path in sorted((root / "flowers" / label).iterdir()):
            try:
                with Image.open(path) as original:
                    image = original.convert("RGB")
                    digest = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()).hexdigest()
                groups[digest].append({"path": path.relative_to(root).as_posix(),
                                       "label": label, "pixel_sha256": digest})
            except (OSError, ValueError, Image.DecompressionBombError):
                invalid.append(path.relative_to(root).as_posix())

    by_class = defaultdict(list)
    duplicates, ambiguous = [], []
    for group in groups.values():
        if len({entry["label"] for entry in group}) != 1:
            ambiguous.extend(entry["path"] for entry in group)
            continue
        by_class[group[0]["label"]].append(group[0])
        duplicates.extend(entry["path"] for entry in group[1:])
    records = []
    rng = random.Random(seed)
    for label in CLASSES:
        examples = sorted(by_class[label], key=lambda x: x["path"])
        if len(examples) < 10:
            raise ValueError(f"Not enough valid images for {label}")
        rng.shuffle(examples)
        n = int(len(examples) * 0.15)
        for split, items in [("validation", examples[:n]), ("test", examples[n:2*n]),
                             ("train", examples[2*n:])]:
            records.extend(dict(item, split=split) for item in items)
    counts = {split: dict(Counter(r["label"] for r in records if r["split"] == split))
              for split in ["train", "validation", "test"]}
    manifest = {
        "source": SOURCE, "archive_sha256": file_hash(archive), "seed": seed,
        "classes": CLASSES, "raw_counts": dict(raw), "split_counts": counts,
        "invalid_files": invalid, "duplicate_files_excluded": duplicates,
        "ambiguous_files_excluded": ambiguous,
        "limitation": "Pixel-identical copies are excluded; near-duplicates and "
                      "same-photo crops may remain. Test data is not an external benchmark.",
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_dir, args.seed)
