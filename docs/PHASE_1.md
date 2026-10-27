# Phase 1 audit and acceptance

## Verified repository state

- Original repository contains README, LICENSE and one training notebook.
- Notebook uses custom ResNet9 with 64x64 RGB input, ToTensor and no normalization.
- Recorded `dataset.classes` is exactly `daisy, dandelion, rose, sunflower, tulip`.
  The README's original 102-category claim is inconsistent with those outputs.
- Notebook saves a raw state_dict to `flowers-resnet9.pth`, but that artifact is
  not committed. Training outputs in a notebook do not restore trained weights.
- RandomCrop is applied to both train and validation datasets. The split is not
  seeded. Inference helper does not explicitly select eval mode. These weaken
  repeatability; historical notebook accuracy is not a serving benchmark.
- No API, deployed UI, RAG, LLM, agent memory or agent workflow exists in the
  original source.

## Added in this phase

- Notebook-compatible inference architecture and strict CPU checkpoint loading.
- Deterministic center-crop serving transform; no GPU or local LLM required.
- JPEG/PNG upload validation, byte/pixel limits, top-three scores and honest 503s
  for missing or incompatible model artifacts.
- Health, readiness, supported labels and prediction endpoints; API test suite.

## Completion gates

- [x] Obtain trusted original checkpoint or retrain/export a replacement.
- [x] Export labels from the SAME training dataset, preserving their index order.
- [x] Load the checkpoint without architecture/key mismatches.
- [x] Predict a real held-out flower image and verify expected label mapping.
- [x] Repeat the same input and confirm identical predictions.
- [x] Evaluate the deterministic transform on held-out images; document accuracy,
      per-class errors and non-flower limitations before making quality claims.

Follow-up: a new checkpoint has now been trained from the user-selected Kaggle data.
All gates above were verified for that baseline; see TRAINING_RESULTS.md. The
original historical checkpoint is still unavailable.
Synthetic checkpoints in tests do not satisfy the real model gates above.

## Subsequent phases

2. Image upload/results UI and measured deployment resource usage.
3. Source-backed care retrieval with citations; unknown information abstention.
4. LangGraph workflow and real tools: identify, retrieve, save and recall plants.
5. User-scoped persistent profiles and conversation state with deletion controls.
6. Opt-in scheduled reminders, failure tests, evaluation and portfolio demo.

Use one framework (LangGraph) initially. Do not add Spring Boot for this Python
learning project unless a concrete integration requires it. RAG content is
knowledge; conversation history and user profiles are separate memory stores.

## Dataset preparation follow-up

The user-selected Kaggle dataset has now been downloaded and audited. See
[DATASET_AUDIT.md](DATASET_AUDIT.md) for class counts and exclusions. Training
scripts now export notebook-compatible weights and label metadata. The original
checkpoint remains unavailable; a newly trained baseline is a separate model.
