"""Train the notebook-compatible ResNet9 and export its exact label mapping."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.inference import preprocess
from app.model import ResNet9


def build_loader(records, data_dir, classes, batch_size, shuffle=False):
    images, targets = [], []
    for entry in records:
        with Image.open(data_dir / entry["path"]) as image:
            images.append(preprocess(image))
        targets.append(classes.index(entry["label"]))
    return DataLoader(TensorDataset(torch.from_numpy(np.stack(images)),
                                   torch.tensor(targets, dtype=torch.long)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model, loader, classes):
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.long)
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            predicted = logits.argmax(1)
            matrix += torch.bincount(labels * len(classes) + predicted,
                                     minlength=len(classes)**2).reshape_as(matrix)
    total = matrix.sum().item()
    return {"accuracy": matrix.diag().sum().item() / total, "loss": total_loss / total,
            "count": total, "confusion_matrix": matrix.tolist(),
            "per_class_recall": {label: matrix[i, i].item() / max(1, matrix[i].sum().item())
                                 for i, label in enumerate(classes)}}


def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    classes = manifest["classes"]
    records = manifest["records"]
    # Defense against accidental train/test overlap in an edited manifest.
    if len({entry["pixel_sha256"] for entry in records}) != len(records):
        raise ValueError("Manifest includes repeated images; rerun data preparation")
    print("Caching preprocessed images; no GPU or external API required.", flush=True)
    loaders = {split: build_loader([r for r in records if r["split"] == split],
                                   args.data_dir, classes, args.batch_size, split == "train")
               for split in ["train", "validation", "test"]}
    model = ResNet9(num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "flowers-resnet9.pth"
    history, best_accuracy = [], -1
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss, count = 0.0, 0
        for batch, (images, labels) in enumerate(loaders["train"], start=1):
            # Horizontal flip augmentation; validation/test remain deterministic.
            flip = torch.rand(len(images)) < 0.5
            images[flip] = images[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            count += len(images)
            if batch % 25 == 0:
                print(json.dumps({"epoch": epoch, "batch": batch,
                                  "of_batches": len(loaders["train"]),
                                  "running_loss": total_loss / count}), flush=True)
        validation = evaluate(model, loaders["validation"], classes)
        result = {"epoch": epoch, "train_loss": total_loss / count,
                  "validation_accuracy": validation["accuracy"],
                  "validation_loss": validation["loss"],
                  "seconds": round(time.monotonic() - epoch_started, 2)}
        history.append(result)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(model.state_dict(), checkpoint)
        print(json.dumps(result), flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    # Test set is evaluated once, after model selection by validation accuracy.
    test_metrics = evaluate(model, loaders["test"], classes)
    (args.output / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    metrics = {
        "source": manifest["source"], "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "architecture": "ResNet9", "torch_version": torch.__version__,
        "seed": args.seed, "split_seed": manifest["seed"], "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "threads": args.threads, "classes": classes,
        "split_counts": dict(Counter(r["split"] for r in records)),
        "history": history, "best_validation_accuracy": best_accuracy,
        "test": test_metrics, "total_seconds": round(time.monotonic() - started, 2),
        "preprocessing": "RGB, shortest-side resize 64 bilinear, center crop 64, divide by 255",
        "limitations": [manifest["limitation"], "Uncalibrated softmax scores",
                        "No unsupported-image rejection or disease diagnosis"],
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
