"""CPU inference. Missing or incompatible checkpoints never produce predictions."""
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def preprocess(image):
    """Deterministic Resize(short side=64), CenterCrop(64), ToTensor equivalent.

    The notebook uses RandomCrop for both training and validation. CenterCrop is
    deliberately used for serving; its accuracy must be evaluated on held-out data.
    No normalization: the original notebook did not normalize its inputs.
    """
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (64, int(64 * height / width))
    else:
        size = (int(64 * width / height), 64)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 64) / 2), round((size[1] - 64) / 2)
    image = image.crop((left, top, left + 64, top + 64))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0


class Predictor:
    def __init__(self, model_path=None, labels_path=None):
        self.model_path = Path(model_path or os.getenv(
            "MODEL_PATH", str(ROOT / "artifacts/flowers-resnet9.pth")))
        self.labels_path = Path(labels_path or os.getenv(
            "LABELS_PATH", str(ROOT / "artifacts/classes.json")))
        self.model = None
        self.checkpoint_sha256 = None
        self.classes = []
        self.error = "Model is not loaded."
        self._lock = Lock()

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        self.model = None
        self.checkpoint_sha256 = None
        if not self.model_path.is_file() or not self.labels_path.is_file():
            self.error = "Trained weights and classes.json are required. See README."
            return
        try:
            import torch
            from app.model import ResNet9

            classes = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if (not isinstance(classes, list) or len(classes) < 2
                    or not all(isinstance(c, str) and c.strip() for c in classes)
                    or len(set(classes)) != len(classes)):
                raise ValueError("classes.json must contain unique, nonempty labels.")
            torch.set_num_threads(1)
            model = ResNet9(num_classes=len(classes))
            # Only load an administrator-provided, trusted state_dict.
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            with self.model_path.open("rb") as source:
                self.checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            self.classes, self.model, self.error = classes, model, None
        except Exception:
            logger.exception("Checkpoint could not be loaded")
            self.error = "Checkpoint could not be loaded. Check server logs and export instructions."

    def predict(self, image):
        if not self.ready:
            raise RuntimeError(self.error)
        import torch

        with self._lock, torch.inference_mode():
            logits = self.model(torch.from_numpy(preprocess(image)).unsqueeze(0))
            if logits.shape != (1, len(self.classes)) or not torch.isfinite(logits).all():
                raise RuntimeError("Model returned invalid output.")
            scores = torch.softmax(logits, dim=1)[0].tolist()
        ranked = sorted(zip(self.classes, scores), key=lambda item: item[1], reverse=True)
        return {
            "predicted_class": ranked[0][0],
            "score": ranked[0][1],
            "top_predictions": [{"label": label, "score": score}
                                for label, score in ranked[:3]],
            "notice": "Scores are uncalibrated model scores, not identification certainty. "
                      "Unsupported flowers and non-flower images may receive high scores.",
        }
