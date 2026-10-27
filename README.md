# Flower Identification & Care AI Agent

An incremental extension of the original PyTorch flower classification notebook.
**Current implementation: Phase 1 prediction API foundation.** Agent workflows,
care retrieval, memory, reminders and web UI are planned, not implemented.

## Current status

The notebook records five categories: **daisy, dandelion, rose, sunflower, tulip**.
The original notebook checkpoint is **not present in this repository**. A new
3-epoch baseline has now been trained: **67.70% held-out test accuracy** on 644
images. See [training results and class-specific limitations](docs/TRAINING_RESULTS.md).
Weights are distributed separately, not committed to Git. The API returns HTTP
503 until valid weights and label metadata are placed in `artifacts/`.

The original notebook is preserved in `deep_learning_project.ipynb`. See
[the source audit and completion gates](docs/PHASE_1.md) and
[the original project description](docs/ORIGINAL_README.md).

## Run locally (Python 3.11 or 3.12 recommended)

Windows PowerShell, from the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs to try the API. CPU-only PyTorch is installed
separately to avoid a CUDA runtime. No paid API key or LLM is needed for Phase 1.
For API-only development without weights, skip the model requirements.

## Train from the selected Kaggle dataset

The data preparation script downloads `alxmamaev/flowers-recognition`, validates
images, removes pixel-identical duplicates and excludes conflicting-label copies.
It writes a reproducible stratified 70/15/15 split. See
[the dataset audit](docs/DATASET_AUDIT.md).

After installing the runtime and model requirements:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.train --epochs 3 --threads 4
```

Training exports weights, labels and metrics to `artifacts/`. The validation set
selects the checkpoint; the held-out test set is evaluated after training.
For longer runs choose `--epochs 10`. Use a separate `--output` folder to retain
an earlier model; training overwrites checkpoint files in the selected folder.
CPU training can be slow. No GPU or API key is required, but dependency/data
downloads need internet. Data and model artifacts stay out of Git.

## Recover/export the trained model

If the original notebook session is still available with its trained `model` and
`dataset`, run this cell **after training**:

```python
from pathlib import Path
import json
import torch

output = Path("artifacts")
output.mkdir(exist_ok=True)
model.eval()
torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
           output / "flowers-resnet9.pth")
(output / "classes.json").write_text(json.dumps(dataset.classes), encoding="utf-8")
```

Download BOTH files from your notebook environment and place them in this
repository's local `artifacts/` directory, then restart the API. Do not export an
untrained model. If the old session/weights no longer exist, the model must be
retrained. Never guess or reorder labels. Only load trusted checkpoint files.

An existing `flowers-resnet9.pth` from this exact training run can also be reused,
but its matching class-index mapping must be recovered and verified.
`MODEL_PATH` and `LABELS_PATH` optionally override the default artifact locations.
Model binaries and private configuration are gitignored.

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Process health and model-ready flag |
| GET | `/ready` | 200 only after successful checkpoint loading; otherwise 503 |
| GET | `/classes` | Labels from the loaded model's classes.json |
| POST | `/predict` | Multipart `file`: JPEG/PNG, at most 5 MiB and 12 million pixels |

Example (Windows: use `curl.exe`):

```bash
curl -X POST http://localhost:8000/predict -F "file=@flower.jpg"
```

Returns a predicted class, uncalibrated softmax score and top-three scores.
A high score does **not** prove the image contains a supported flower. No
out-of-distribution detector or calibrated rejection threshold is implemented.
Serving uses deterministic Resize(short side=64), CenterCrop(64), ToTensor,
with no normalization. This deliberately replaces the notebook's random crop
and requires held-out evaluation before quality claims.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover missing artifacts, invalid uploads, deterministic preprocessing,
synthetic checkpoint/label wiring and incompatible weights. Model tests skip if
PyTorch is not installed. Synthetic checkpoints test infrastructure, not accuracy.

## Next milestones

1. Improve baseline quality, especially rose/tulip separation; add independent evaluation.
2. Add image upload/results UI and measure hosting memory requirements.
3. Add source-backed care information and a LangGraph tool workflow.
4. Add user-scoped persistent memory and opt-in reminders.

This is a local-development API foundation. Public deployment needs gateway
request limits, authentication/rate limiting appropriate to use, and measured
resource usage. No hosting or Microsoft integration has been provisioned.

## License

MIT. See [LICENSE](LICENSE).
