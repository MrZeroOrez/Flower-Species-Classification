"""Exercise the exported checkpoint through the real upload API on a held-out image."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def verify():
    manifest = json.loads(Path("data/manifest.json").read_text())
    sample = next(row for row in manifest["records"] if row["split"] == "test")
    image = Path("data") / sample["path"]
    with TestClient(create_app()) as client:
        assert client.get("/ready").status_code == 200, "Checkpoint failed to load"
        assert client.get("/classes").json()["classes"] == manifest["classes"]
        predictions = []
        for _ in range(2):
            with image.open("rb") as source:
                response = client.post("/predict", files={"file": (image.name, source, "image/jpeg")})
            assert response.status_code == 200, response.text
            predictions.append(response.json())
        assert predictions[0] == predictions[1], "Inference was not repeatable"
    result = {"sample": sample["path"], "split": "test", "expected": sample["label"],
              "predicted": predictions[0]["predicted_class"],
              "correct": predictions[0]["predicted_class"] == sample["label"],
              "http_status": 200, "repeatable": True, "response": predictions[0]}
    Path("artifacts/smoke_prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
