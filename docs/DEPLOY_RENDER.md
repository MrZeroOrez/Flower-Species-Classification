# Render Free deployment

Prepared target: one Python **Free** web service, serving the UI, prediction and
care workflow together. No database, persistent disk, paid worker or LLM provider
is needed for this version. The baseline service was verified on 2026-10-28:
https://flower-species-classification-pp6o.onrender.com/
Readiness, a real-image prediction and source-backed care returned HTTP 200.
This confirms serving/integration, not flower identification accuracy.

## Cost conditions checked on 2026-10-28

Official references:
- https://render.com/docs/free
- https://render.com/docs/compute-plans
- https://render.com/docs/blueprint-spec

Choose the free workspace tier and **Free** service compute. For strict zero
cash-spend operation, use an account/workspace with no payment method. The free
service has 512 MB RAM and shares 750 monthly free instance hours with other free
services in the workspace. It sleeps after 15 idle minutes; waking can take about
a minute. It is intended for demos/hobby use, not an always-on production SLA.

Free services also consume bandwidth and build allowances. Render's documentation
says excess bandwidth/build usage can be billed when a payment method is present;
without one, service/build suspension applies instead. Therefore **Free does not
mean an unconditional no-charge guarantee for an account with a saved card**.
Do not upgrade compute, add a disk, attach a paid database or enable paid LLM calls.
This config cannot control your account's saved payment methods or unrelated costs.

`render.yaml` explicitly selects `plan: free`, disables automatic deploys to reduce
unnecessary builds, and sets `DISABLE_LLM=true`. That switch overrides any existing
LLM credentials. Keep it enabled for this deployment. The optional LLM package is
not installed by the build script.

## Model package: one-time publishing step

The model is not in Git. Use the separately delivered, unchanged
`Flower_Classifier_Baseline_Model.zip`. Publish that ZIP as a public GitHub release
asset on the same public repository, then copy the asset's public download URL.
Do not use a ChatGPT Library/download URL: the build cannot authenticate to it.

1. Open the repository's Releases page and draft a release, for example `model-v1`.
2. Attach the exact model ZIP, then publish the release.
3. Copy the ZIP download link. Set it as `MODEL_PACKAGE_URL` in Render.

The baseline is published at
https://github.com/MrZeroOrez/Flower-Species-Classification/releases/tag/model-v1.
Do not repack or modify a published ZIP;
the downloader verifies the exact package and artifact SHA-256 hashes. It extracts
only the model and label metadata and fails the build on mismatches. Approved
older packages remain supported so updating source does not require an immediate
model replacement. When `MODEL_PACKAGE_URL` is set it takes precedence over
cached local weights, including during a model upgrade.

If a connected deployment tool can supply these exact model files in its build
source, the script can also reuse them after hash verification. No retraining runs
on Render. Files created during the build belong to the deployed build artifact;
runtime uploads and local memory must not be treated as durable storage.

## Create the service

Use the connected Render integration after account connection, or in the dashboard
choose **New → Web Service**, connect this repository and apply these values:

| Setting | Value |
|---|---|
| Repository | `MrZeroOrez/Flower-Species-Classification` |
| Branch | `feature/flower-agent-phase-1` (contains the unmerged implementation) |
| Runtime | Python |
| Compute plan | **Free** |
| Build command | `bash scripts/render_build.sh` |
| Start command | `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 8` |
| Health check | `/ready` |
| Auto-deploy | Off/manual |
| `PYTHON_VERSION` | `3.12.10` |
| `DISABLE_LLM` | `true` |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `MODEL_PACKAGE_URL` | Public HTTPS URL of the exact model ZIP |

Alternatively create a Blueprint using `render.yaml`. Review the proposed service
plan before creating anything. This branch is explicit because main still contains
the original notebook. After merging, update the deployment branch deliberately.

## Verification and resource evidence

Local checks passed:
- 47 API/data/care/deployment/training/package tests.
- Offline import of the exact delivered ZIP, hash checks and checkpoint loading.
- Real-image prediction and no-key care endpoint: HTTP 200.
- Initial prediction/care process peak RSS: about 375 MiB.
- 12-megapixel synthetic JPEG after decode optimization: about 427 MiB peak RSS.

These are local Linux measurements, not proof of Render RAM/latency behavior.
Decode plus inference is serialized; phone orientation handling remains covered.
One worker avoids duplicate model copies. The server bounds concurrent requests;
busy requests may receive 503 and should be retried. Exact memory needs can vary.

After deploying, verify `/ready` is 200, open `/`, upload a real flower image,
check top-three scores and source-linked care, then inspect actual service memory
and logs. Verify both a cold start and recovery after idle sleep. Do not claim the
deployment is complete until the public URL passes these checks.

Persistent conversations, accounts and plant memory are not implemented. Free
Render local files would not provide durable memory; choose storage separately
when that phase is built. No card or paid resource is needed for the current demo.

## Upgrade to the v2 rose-focused candidate

The v2 package improves rose recall but reduces tulip recall; read
[the measured trade-offs](TRAINING_RESULTS.md) before switching. The live service
was verified with v1; v2 has been verified locally but is not yet published/deployed.

1. Create a new GitHub release with tag `model-v2`, targeting the updated
   `feature/flower-agent-phase-1` branch. Attach the unchanged
   `Flower_Classifier_Improved_Model_v2.zip`. Preserve the v1 release.
2. Copy its ZIP asset download URL into Render's `MODEL_PACKAGE_URL`. Keep all
   other free-service settings and `DISABLE_LLM=true`.
3. Manually deploy the latest branch commit. The URL takes priority over cached
   old weights, so a normal deployment is sufficient.
4. Check `/health`: `model_ready` should be true and `checkpoint_sha256` should be
   `6ad46d27975a4ba1430f316a9127bb13abfced63837287b89823a7960ab66be2`.
   Then retry independent rose **and** tulip photos.

To roll back, restore the v1 asset URL and manually redeploy. Both package hashes
remain approved; no retraining or extra cloud service is needed on Render.
