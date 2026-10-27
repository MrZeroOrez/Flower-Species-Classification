# Initial trained baseline

Source: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

Trained for 3 epochs on CPU in 495.2 seconds.

- Best validation accuracy: 66.30% on 644 images.
- Held-out test accuracy: 67.70% on 644 images.
- API readiness and upload prediction returned HTTP 200 with the exported model.
- Held-out smoke image expected `daisy`; predicted `daisy`.
- Repeated API predictions on the same image were identical.

| Class | Test recall |
|---|---:|
| daisy | 73.68% |
| dandelion | 90.45% |
| rose | 28.21% |
| sunflower | 56.88% |
| tulip | 78.23% |

This is an initial baseline, not a production-quality classifier. Rose recall is
particularly weak and many roses are confused with tulips. Overall accuracy
must not be presented as identification certainty. Same-dataset held-out results
do not measure performance on arbitrary real-world photos. Near-duplicates may
remain despite exact pixel deduplication. No disease diagnosis or non-flower
rejection is implemented.

The checkpoint and exact labels are exported separately from Git. The original
historical notebook weights were not recovered. This is a newly trained model.

## Verification

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_model
```

The full metrics JSON includes the confusion matrix, split and checkpoint hashes,
hyperparameters and epoch history. The model package includes the split manifest
and source attribution, but no dataset images.

## Fine-tuned model v2 candidate (2026-10-28)

The deployed baseline confused 58 of 117 test roses with tulips. A six-epoch CPU
fine-tuning run started from that three-epoch checkpoint. No test images were used
for gradient updates. The checkpoint was selected by **validation macro recall**
(equal importance for each class); epoch 5 was selected. Test evaluation followed
selection. The same 644-image benchmark has now been reused for model comparison;
it is not independent external validation. The user's three rose photos were not
available for evaluation.

| Metric | Baseline | v2 candidate |
|---|---:|---:|
| Overall test accuracy | 67.70% | 79.19% |
| daisy test recall | 73.68% | 75.44% |
| dandelion test recall | 90.45% | 85.35% |
| rose test recall | 28.21% | 75.21% |
| sunflower test recall | 56.88% | 89.91% |
| tulip test recall | 78.23% | 70.75% |
| Rose → tulip errors (117 actual roses) | 58 | 12 |
| Tulip → rose errors (147 actual tulips) | 5 | 23 |

**Trade-off:** rose recall improves strongly, but tulip recall falls from 78.23%
to 70.75%; dandelion recall also falls. This is an improved rose-focused demo
candidate, not a complete solution to rose/tulip confusion. Check both categories
on new real photos before treating it as an upgrade for a particular use case.

Training: inverse-frequency class-weighted cross entropy, label smoothing 0.05,
Adam starting at 0.0003 with cosine decay, batch 32, seed 42, four CPU threads,
deterministic serving preprocessing and horizontal-flip training augmentation.
Fine-tuning and evaluation took 828.33 seconds. Architecture,
input size, label order and serving memory requirements remain unchanged.

The API regression example is a test-set rose that the baseline calls tulip and
v2 calls rose. Its HTTP 200 response matches direct inference. This is a selected
regression example; the aggregate table above represents all test images.

Full metrics and confusion matrix: [model-v2-metrics.json](model-v2-metrics.json).

- Model SHA-256: `6ad46d27975a4ba1430f316a9127bb13abfced63837287b89823a7960ab66be2`
- Package SHA-256: `243399685dd13082235a2a8c18ebb03577a4d00ba9d1c7fe00391a44f2ba8aad`

The package is `Flower_Classifier_Improved_Model_v2.zip`. It is a separate immutable
release candidate; publishing and setting its URL on Render are still required.
The existing `model-v1` package remains accepted for rollback. `/health` now reports
the loaded checkpoint SHA-256 so the active model can be checked after deployment.
