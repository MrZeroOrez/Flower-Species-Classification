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
