"""Export an evaluated checkpoint without putting weights or dataset images in Git."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package_model(model_dir, manifest, output):
    model_dir, manifest, output = Path(model_dir), Path(manifest), Path(output)
    checkpoint = model_dir / "flowers-resnet9.pth"
    metrics = json.loads((model_dir / "metrics.json").read_text())
    classes = json.loads((model_dir / "classes.json").read_text())
    if sha256(checkpoint) != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the evaluated metrics")
    if sha256(manifest) != metrics["manifest_sha256"]:
        raise ValueError("Manifest does not match the evaluated data split")
    if classes != metrics["classes"]:
        raise ValueError("Class order does not match the evaluated model")
    if output.exists():
        raise FileExistsError("Choose a new package name; published packages must remain immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "Flower Lab evaluated ResNet9 model\n\n"
        "Place artifacts/flowers-resnet9.pth and classes.json in the application's artifacts folder.\n"
        "Read metrics.json for evaluation and training provenance. No dataset images are included.\n"
        "Scores are uncalibrated; unsupported images may score highly.\n"
        "This is a same-dataset comparison, not an independent real-world accuracy guarantee.\n"
        "Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition\n"
    )
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in ["flowers-resnet9.pth", "classes.json", "metrics.json"]:
            archive.write(model_dir / name, "artifacts/" + name)
        archive.write(manifest, "artifacts/dataset_manifest.json")
        archive.writestr("MODEL_README.txt", readme)
    result = {"path": str(output.resolve()), "package_sha256": sha256(output),
              "artifacts": {name: sha256(model_dir / name)
                            for name in ["flowers-resnet9.pth", "classes.json"]}}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package_model(args.model_dir, args.manifest, args.output)
