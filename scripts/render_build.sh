#!/usr/bin/env bash
set -euo pipefail
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -r requirements-model.txt
python -m scripts.fetch_model
python -c 'from app.inference import Predictor; p = Predictor(); p.load(); assert p.ready, p.error; print("Model ready for deployment")'
