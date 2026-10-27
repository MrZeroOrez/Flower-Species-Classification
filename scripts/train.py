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
    recall = {label: matrix[i, i].item() / max(1, matrix[i].sum().item())
              for i, label in enumerate(classes)}
    return {"accuracy": matrix.diag().sum().item() / total, "loss": total_loss / total,
            "count": total, "confusion_matrix": matrix.tolist(),
            "macro_recall": sum(recall.values()) / len(classes),
            "per_class_recall": recall}


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, threads and learning-rate must be positive")
    if not 0 <= args.label_smoothing <= 1:
        raise ValueError("label-smoothing must be between 0 and 1")
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
    initial_sha = None
    if args.resume:
        if args.resume.resolve() == (args.output / "flowers-resnet9.pth").resolve():
            raise ValueError("Use a separate output directory to preserve the original checkpoint")
        resume_classes = json.loads(args.resume.with_name("classes.json").read_text())
        if resume_classes != classes:
            raise ValueError("Resume checkpoint class order differs from the manifest")
        initial_sha = hashlib.sha256(args.resume.read_bytes()).hexdigest()
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 20)
        if args.cosine else None)
    train_counts = Counter(r["label"] for r in records if r["split"] == "train")
    weights = (torch.tensor([sum(train_counts.values()) / (len(classes) * train_counts[c])
                             for c in classes]) if args.balanced_loss else None)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "flowers-resnet9.pth"
    history, best_score, best_epoch = [], -1, None
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss, count = 0.0, 0
        for batch, (images, labels) in enumerate(loaders["train"], start=1):
            # Horizontal flip augmentation; validation/test remain deterministic.
            flip = torch.rand(len(images)) < 0.5
            images[flip] = images[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
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
                  "validation_macro_recall": validation["macro_recall"],
                  "validation_per_class_recall": validation["per_class_recall"],
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": round(time.monotonic() - epoch_started, 2)}
        history.append(result)
        score = validation[args.selection_metric]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), checkpoint)
        if scheduler:
            scheduler.step()
        print(json.dumps(result), flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    # Test set is evaluated once, after selection using the chosen validation metric.
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
        "history": history, "best_epoch": best_epoch,
        "selection_metric": args.selection_metric, "best_selection_score": best_score,
        "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
        "initial_checkpoint_sha256": initial_sha,
        "balanced_loss": args.balanced_loss, "label_smoothing": args.label_smoothing,
        "cosine_schedule": args.cosine,
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
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint; classes.json must be adjacent")
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_recall"], default="accuracy")
    train(parser.parse_args())
