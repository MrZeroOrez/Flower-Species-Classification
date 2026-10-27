from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import Predictor, preprocess
from app.main import create_app


def image_bytes(fmt="PNG"):
    output = BytesIO()
    Image.new("RGB", (100, 80), "yellow").save(output, format=fmt)
    return output.getvalue()


@pytest.fixture
def unavailable(tmp_path):
    predictor = Predictor(tmp_path / "absent.pth", tmp_path / "absent.json")
    with TestClient(create_app(predictor)) as client:
        yield client


def test_missing_model_is_honest(unavailable):
    assert unavailable.get("/health").json() == {"status": "ok", "model_ready": False}
    assert unavailable.get("/ready").status_code == 503
    assert unavailable.get("/classes").status_code == 503
    response = unavailable.post("/predict", files={"file": ("flower.png", image_bytes())})
    assert response.status_code == 503
    assert "predicted_class" not in response.json()


def test_ui_and_assets_available_without_model(unavailable):
    home = unavailable.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers["content-type"]
    assert 'id="image-file"' in home.text
    for asset in ["app.js", "styles.css"]:
        assert unavailable.get("/static/" + asset).status_code == 200
    assert unavailable.get("/static/flowers-resnet9.pth").status_code == 404


@pytest.mark.parametrize("data,status", [(b"", 400), (b"not an image", 400),
                                           (b"x" * (5 * 1024 * 1024 + 1), 413)])
def test_invalid_uploads(unavailable, data, status):
    assert unavailable.post("/predict", files={"file": ("x.png", data)}).status_code == status


def test_format_checked_from_bytes(unavailable):
    response = unavailable.post("/predict", files={"file": ("fake.png", image_bytes("GIF"))})
    assert response.status_code == 415


def test_pixel_limit(unavailable, monkeypatch):
    monkeypatch.setattr("app.main.MAX_PIXELS", 100)
    response = unavailable.post("/predict", files={"file": ("large.png", image_bytes())})
    assert response.status_code == 413


def test_preprocessing_is_deterministic():
    import numpy as np
    image = Image.new("RGBA", (150, 90), (255, 100, 0, 120))
    first = preprocess(image)
    assert first.shape == (3, 64, 64)
    assert first.dtype == np.float32
    assert 0 <= first.min() <= first.max() <= 1
    np.testing.assert_array_equal(first, preprocess(image))


def test_checkpoint_round_trip(tmp_path):
    """Synthetic weights test wiring and label order, NOT real classification accuracy."""
    import hashlib
    import json
    torch = pytest.importorskip("torch")
    from app.model import ResNet9

    torch.set_num_threads(1)
    model = ResNet9(num_classes=2)
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([-5.0, 5.0]))
    checkpoint = tmp_path / "test.pth"
    labels = tmp_path / "classes.json"
    torch.save(model.state_dict(), checkpoint)
    labels.write_text(json.dumps(["first", "second"]))
    with TestClient(create_app(Predictor(checkpoint, labels))) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").json()["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = client.post("/predict", files={"file": ("x.png", image_bytes())})
        assert result.status_code == 200
        assert result.json()["predicted_class"] == "second"
        assert result.json()["score"] > 0.99
        assert result.json()["top_predictions"][0]["label"] == "second"
        repeated = client.post("/predict", files={"file": ("x.png", image_bytes())})
        assert repeated.json() == result.json()


def test_incompatible_checkpoint_is_unavailable(tmp_path):
    torch = pytest.importorskip("torch")
    weights, labels = tmp_path / "bad.pth", tmp_path / "classes.json"
    torch.save({"wrong_layer": torch.zeros(1)}, weights)
    labels.write_text('["daisy", "rose"]')
    with TestClient(create_app(Predictor(weights, labels))) as client:
        assert client.get("/ready").status_code == 503
