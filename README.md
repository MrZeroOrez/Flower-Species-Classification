# Flower Identification & Care AI Agent

A Python application that identifies five flower categories from photos and provides
source-backed care guidance through a bounded LangGraph tool workflow.

**[Live demo](https://flower-species-classification-pp6o.onrender.com/)** ·
**[API documentation](https://flower-species-classification-pp6o.onrender.com/docs)** ·
**[Download v2 model](https://github.com/MrZeroOrez/Flower-Species-Classification/releases/tag/model-v2)**

The application code is merged into **`main`**. The deployed service was checked on
2026-10-28: `/health` returned HTTP 200, `model_ready: true`, and the exact v2
checkpoint hash. Free hosting can take time to wake after inactivity.

## Features

- Upload or drag in a JPEG/PNG image, preview it, and compare the top three model scores.
- Identify **daisy, dandelion, rose, sunflower and tulip** using a CPU PyTorch ResNet9.
- Explore care references for sunlight, soil, watering, pruning and seasons, with source links.
- Run a bounded LangGraph workflow that selects a typed `lookup_flower_care` tool and returns reviewed evidence or an honest clarification/abstention.
- Run without an LLM API key. Optional remote LLM tool routing is available separately.
- Validate uploads, preserve phone-photo orientation, and handle unavailable models and stale requests.
- Download approved model packages with SHA-256 verification and check the loaded checkpoint through `/health`.

Uploaded photos are not saved by the application. Care guidance also works without
classifier weights; users can select or correct the flower category manually.

## Technology

| Component | Implementation |
|---|---|
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| Image classifier | PyTorch ResNet9, NumPy, Pillow; CPU inference |
| Agent workflow | LangGraph and a typed care-lookup tool |
| Knowledge | Curated structured profiles with RHS and NC State Extension citations |
| Interface | HTML, CSS and JavaScript served by FastAPI |
| Verification | pytest, FastAPI TestClient; optional Playwright browser checks |
| Deployment | One Render Free Python web service; separately published model assets |

The no-key care mode is labelled **Guided source lookup**. It retrieves curated
facts; it is not free-form LLM chat or vector RAG. Saved conversation memory,
plant profiles, reminders and Microsoft 365 integration are not implemented.

## Model results and limitations

The v2 model fine-tunes the original three-epoch baseline for six additional CPU
epochs. The checkpoint was selected at fine-tuning epoch 5 using validation macro
recall, giving each class equal importance. The same 644-image test split was used
for comparison; these results are not an independent real-world benchmark.

| Metric | Baseline v1 | Deployed v2 |
|---|---:|---:|
| Overall test accuracy | 67.70% | 79.19% |
| Daisy recall | 73.68% | 75.44% |
| Dandelion recall | 90.45% | 85.35% |
| Rose recall | 28.21% | 75.21% |
| Sunflower recall | 56.88% | 89.91% |
| Tulip recall | 78.23% | 70.75% |
| Rose → tulip errors, out of 117 roses | 58 | 12 |
| Tulip → rose errors, out of 147 tulips | 5 | 23 |

**Trade-off:** v2 improves rose recognition and overall accuracy, while tulip and
dandelion recall decline. Rose/tulip separation remains imperfect. A displayed
score is an **uncalibrated model score**, not identification certainty. Unsupported
flowers and non-flower images can still receive high scores. Confirm the flower
before following its care guidance. No disease diagnosis or unsupported-image
rejection is implemented.

See [training results](docs/TRAINING_RESULTS.md),
[full v2 metrics and confusion matrix](docs/model-v2-metrics.json), and
[dataset audit](docs/DATASET_AUDIT.md). Exact pixel duplicates were removed, but
near-duplicates or crops of the same photograph may remain across splits.

## Run locally

Use Python 3.11 or 3.12. Clone the merged application:

```bash
git clone https://github.com/MrZeroOrez/Flower-Species-Classification.git
cd Flower-Species-Classification
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
$env:DISABLE_LLM = "true"
$env:MODEL_PACKAGE_URL = "https://github.com/MrZeroOrez/Flower-Species-Classification/releases/download/model-v2/Flower_Classifier_Improved_Model_v2.zip"
.\.venv\Scripts\python.exe -m scripts.fetch_model
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Linux / macOS shell

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-model.txt
export DISABLE_LLM=true
export MODEL_PACKAGE_URL="https://github.com/MrZeroOrez/Flower-Species-Classification/releases/download/model-v2/Flower_Classifier_Improved_Model_v2.zip"
.venv/bin/python -m scripts.fetch_model
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

CPU wheel availability depends on platform; unsupported platforms may need their
platform-specific PyTorch installation. Open http://127.0.0.1:8000 or
http://127.0.0.1:8000/docs. No Node runtime or frontend build is needed to serve the
app. Dependencies and model downloads require internet access.

For API/care development without a classifier, omit the model requirements and
model download. Prediction and readiness return HTTP 503 until valid weights and
matching labels are available. `MODEL_PATH` and `LABELS_PATH` can override their
locations. Only load trusted checkpoints; never guess or reorder labels.

## API

| Method | Path | Behavior |
|---|---|---|
| GET | `/` | Upload interface and care panel |
| GET | `/health` | Process health, model readiness and loaded checkpoint SHA-256 |
| GET | `/ready` | HTTP 200 when the model is loaded; otherwise 503 |
| GET | `/classes` | Class-index mapping from the loaded model |
| POST | `/predict` | Multipart field `file`; JPEG/PNG, at most 5 MiB and 12 million pixels |
| POST | `/care/chat` | JSON `flower` and optional `question`; reviewed care facts and citations |
| GET | `/docs` | Interactive OpenAPI documentation |

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@flower.jpg"
curl -X POST http://127.0.0.1:8000/care/chat -H "Content-Type: application/json" -d '{"flower":"rose","question":"How should I water it?"}'
```

Use `curl.exe` in Windows PowerShell; the care JSON example above uses POSIX shell
quoting. Serving preprocessing is RGB, shortest-side resize to 64 pixels, centre
crop to 64 × 64, then division by 255, without additional normalization.

## Care workflow and optional LLM routing

Care references cover the five categories, with species/cultivar caveats. Unknown
or uncovered questions receive a clarification or abstention. The tool returns
reviewed facts and citations rather than generating botanical claims.

Optional LLM routing requires `requirements-llm.txt` and all three server variables:
`LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`. It makes at most one bounded call to
select validated tool arguments; failures fall back to guided lookup with a notice.
This optional integration has been tested with fake models, not a live provider.
**`DISABLE_LLM=true` overrides credentials and prevents provider calls.**

See [workflow, sources and configuration](docs/CARE_WORKFLOW.md).

## Training and model export

Dataset: [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).
Preparation audits 4,317 images, excludes two duplicate copies and eight
conflicting-label images, and creates seeded splits of 3,019 training, 644
validation and 644 test images.

Run these modules with the Python executable from your virtual environment:

```bash
python -m scripts.prepare_data
python -m scripts.train --epochs 3 --threads 4
python -m scripts.train --resume artifacts/flowers-resnet9.pth --output training-output/rose-improvement --epochs 6 --learning-rate 0.0003 --balanced-loss --label-smoothing 0.05 --cosine --selection-metric macro_recall --threads 4
python -m scripts.package_model --model-dir training-output/rose-improvement --output Flower_Classifier_Improved_Model_v2.zip
```

The second training command reproduces the fine-tuning configuration: balanced
loss, label smoothing and cosine learning-rate decay. It requires the baseline
checkpoint and adjacent `classes.json`. Use a separate output directory to retain
earlier weights. Test evaluation follows validation-based checkpoint selection.
Training is CPU-intensive and is not run on Render during deployment.

The exporter verifies checkpoint, labels and dataset-split provenance against the
metrics. It refuses to overwrite an existing ZIP. Publish a newly trained model
under a new package name, approve its exact checksums in `scripts/fetch_model.py`,
and update the deployment URL deliberately. Do not replace an immutable release
asset with different bytes. Weights, dataset images and private configuration are
excluded from Git.

The historical notebook remains in `deep_learning_project.ipynb`; its original
weights were not recovered. See [original project description](docs/ORIGINAL_README.md)
and [initial implementation audit](docs/PHASE_1.md).

## Verification

**47 Python tests passed** for the merged implementation. They cover API/upload
validation, deterministic inference and label wiring, care evidence and abstention,
tool-routing fallbacks, concurrency, per-class evaluation, package provenance and
model-cache upgrade behavior. Synthetic checkpoints test wiring, not accuracy.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Model tests skip if PyTorch is unavailable. Optional browser checks require Node,
the dataset manifest/images, and trained artifacts:

```bash
npm ci
npx playwright install chromium
python -m scripts.check_browser
```

The browser runner starts a temporary local API on port 8011 and stops it after
checking the UI. Prior end-to-end checks covered real-image inference, reset/error
states, care citations/abstention and mobile layout. V2 was separately verified by
package import, checkpoint loading and a real-image API regression check. See
[UI verification](docs/UI_VERIFICATION.md).

## Deploy on Render Free

Use the **`main`** branch for the merged application. For an existing service,
change **Settings → Build & Deploy → Branch** to `main` before the next deployment.
The existing blueprint and older setup notes name the original feature branch;
select `main` explicitly when using those instructions.

- Runtime: Python; compute: **Free**; root directory: blank.
- Build: `bash scripts/render_build.sh`
- Start: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 8`
- Health check: `/ready`; deployments: manual.
- Environment: `PYTHON_VERSION=3.12.10`, `DISABLE_LLM=true`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Set `MODEL_PACKAGE_URL` to the v2 ZIP asset download URL shown in the local setup.

After deploying, `/health` should report this v2 checkpoint SHA-256:

```text
6ad46d27975a4ba1430f316a9127bb13abfced63837287b89823a7960ab66be2
```

Both v1 and v2 package hashes remain approved. To roll back the model, restore the
v1 release-asset URL and manually redeploy. No extra hosting service is required.

Free hosting has idle sleep and shared usage limits. Strict zero cash spend
requires free resources and no saved payment method; accounts with a payment
method can incur usage overages. See [deployment instructions and billing
conditions](docs/DEPLOY_RENDER.md) for details and local memory measurements.
This is a demonstration application with imperfect recognition; broader public
use also needs appropriate request controls and measured resource usage.

## Future work

1. Evaluate independent real photos and improve rose/tulip separation without hiding class-level regressions.
2. Add and evaluate unsupported-image rejection and score calibration.
3. Verify optional LLM routing with a live provider and expand reviewed care coverage.
4. Add user-scoped persistent memory, plant profiles and opt-in reminders.
5. Explore vector RAG and Microsoft 365 integration as separate features.

## License

MIT. See [LICENSE](LICENSE).
