# Kaggle dataset audit

Source: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

Archive SHA-256: `eb9a26fc06854affdd50ea1f7987ab6676f22f6f80c3a71f175249fdc8e888a3`

Downloaded archive: 4,317 images; five categories.

| Class | Raw | Train | Validation | Test |
|---|---:|---:|---:|---:|
| daisy | 764 | 536 | 114 | 114 |
| dandelion | 1052 | 738 | 157 | 157 |
| rose | 784 | 546 | 117 | 117 |
| sunflower | 733 | 513 | 109 | 109 |
| tulip | 984 | 686 | 147 | 147 |

- Invalid images: 0
- Duplicate copies excluded: 2
- Conflicting-label images excluded: 8

Seed: 42. Stratified split: approximately 70% train, 15% validation, 15% test.
Decoded RGB pixel hashes prevent exact-copy leakage across splits. Near-duplicates
and different crops of the same photograph may remain; this is not an external benchmark.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.train --epochs 10 --threads 4
```

Use `--epochs 3` for the initial baseline. Models are selected by validation accuracy.
The test set is evaluated after model selection. Do not tune repeatedly on test scores.
Training/export uses the exact class order and deterministic serving preprocessing.
Reproducibility is expected within the same software/hardware environment; different
PyTorch versions and hardware may change numerical results.

Data, model binaries and local training logs are excluded from Git.
Review the dataset source terms before redistributing images.

## Next quality work

- Check perceptual near-duplicates and use an independent image collection.
- Compare transfer learning and larger image sizes if baseline quality is inadequate.
- Measure per-class errors, score calibration and unsupported-image rejection.
- Measure inference RAM and latency before selecting a free hosting instance.
