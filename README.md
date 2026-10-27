# Flower Identification & Care AI Agent

An incremental extension of the original PyTorch flower classification notebook.
**Current implementation: prediction API, trained baseline, upload UI and source-backed
LangGraph care workflow.** Optional LLM tool routing is implemented but not live-provider
verified. Saved memory, vector retrieval and reminders are not implemented.

## Current status

The notebook records five categories: **daisy, dandelion, rose, sunflower, tulip**.
The original notebook checkpoint is **not present in this repository**. A new
3-epoch baseline has now been trained: **67.70% held-out test accuracy** on 644
images. A fine-tuned **v2 candidate reaches 79.19% overall test accuracy and 75.21% rose recall**
(up from 28.21%), with tulip recall falling to 70.75% (from 78.23%). The live app
continues to use v1 until its package URL is upgraded. See
[training results, confusion counts and trade-offs](docs/TRAINING_RESULTS.md).
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

Open http://localhost:8000 for the image upload UI, or http://localhost:8000/docs
to try the API directly. CPU-only PyTorch is installed
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

To fine-tune an existing evaluated checkpoint while preserving the original files:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --resume artifacts/flowers-resnet9.pth --output training-output/rose-improvement --epochs 6 --learning-rate 0.0003 --balanced-loss --label-smoothing 0.05 --cosine --selection-metric macro_recall --threads 4
```

This run uses inverse class-frequency loss weights, a decreasing learning rate,
and mean per-class validation recall for checkpoint selection. The test set is
evaluated after selection. `--resume` requires adjacent `classes.json` in the same
order and refuses to overwrite the input checkpoint. The original split stays fixed;
reusing this benchmark for comparisons does not make it independent validation.

Export the resulting evaluated checkpoint with:

```powershell
.\.venv\Scripts\python.exe -m scripts.package_model --model-dir training-output/rose-improvement --output Flower_Classifier_Improved_Model_v2.zip
```

The exporter checks checkpoint, label order and split provenance against the
metrics. It refuses to overwrite an existing ZIP. A new package must be reviewed
and its exact checksums approved in `scripts/fetch_model.py` before deploying it.

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

## Image upload UI

The home page is served by the same FastAPI process, using local HTML/CSS/JS.
No Node runtime, frontend build, CDN or extra hosting service is required to run it.
Choose or drag in one JPEG/PNG image, inspect its preview, and select **Identify
flower**. Results show the top class and three model scores. Scores are explicitly
uncalibrated; the baseline quality limitations stay visible.

The UI handles empty/invalid/oversized images, missing model artifacts, network
failures, server errors and request timeouts. Replacing or clearing an image
cancels the old request, preventing stale results. The API does not save uploads.

Optional browser checks (Node.js required only for this verification):

```powershell
npm install
npx playwright install chromium
.\.venv\Scripts\python.exe -m scripts.check_browser
```

These checks require `data/manifest.json`, dataset images and trained artifacts
from the earlier preparation/training steps. The runner temporarily starts the
API on port 8011 and stops it after checking the browser flow. Screenshots and
logs are local, ignored QA files. See [UI verification](docs/UI_VERIFICATION.md).

## Flower care and LangGraph

The home page includes a care panel with source-linked answers for the five
categories. A prediction preselects a category, which you can correct manually.
Select **Show care guide**, or ask about light, soil, watering, pruning or season.
References also work without classifier weights. Unknown or uncovered questions
receive a clarification/refusal instead of invented care instructions.

The default **Guided source lookup** uses no LLM key. An optional remote model can
select a validated lookup tool; it never writes the botanical facts or URLs. See
[care workflow, sources and optional configuration](docs/CARE_WORKFLOW.md).
This does not yet add saved conversations, plant profiles, reminders or vector RAG.

## Deploy the demo on Render Free

A one-service deployment blueprint, CPU-only build command, checksum-verified model
package downloader and LLM kill switch are included. See
[Render Free setup and billing conditions](docs/DEPLOY_RENDER.md).
[Live Flower Lab](https://flower-species-classification-pp6o.onrender.com/) has
been verified with readiness, prediction and source-backed care requests. Serving
success does not establish identification quality. Free hosting has sleep and usage limits;
strict zero spend requires free resources and no payment method on the account.

## Next milestones

1. Validate the v2 candidate on independent photos and improve remaining rose/tulip confusion.
2. Broaden independent real-photo evaluation while staying within the hosting memory limit.
3. Verify optional routing with a real configured LLM; expand reviewed knowledge and evaluate answers.
4. Add user-scoped persistent memory and opt-in reminders.

This is a demonstration application. Public deployment needs gateway
request limits, authentication/rate limiting appropriate to use, and measured
resource usage. Microsoft integration has not been provisioned.

## License

MIT. See [LICENSE](LICENSE).
