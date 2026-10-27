import hashlib
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import create_app, decode_image
from app.care import make_chat_model
from scripts import fetch_model


def package_fixture(tmp_path, monkeypatch, expected_weight=b"model"):
    archive = tmp_path / "model.zip"
    with ZipFile(archive, "w") as package:
        package.writestr("artifacts/flowers-resnet9.pth", b"model")
        package.writestr("artifacts/classes.json", b'["daisy"]')
    monkeypatch.setattr(fetch_model, "PACKAGE_SHA256", fetch_model.digest(archive))
    monkeypatch.setattr(fetch_model, "ARTIFACTS", {
        "flowers-resnet9.pth": hashlib.sha256(expected_weight).hexdigest(),
        "classes.json": hashlib.sha256(b'["daisy"]').hexdigest(),
    })
    return archive


def test_model_package_import_verifies_both_files(tmp_path, monkeypatch):
    archive = package_fixture(tmp_path, monkeypatch)
    output = tmp_path / "artifacts"
    fetch_model.install_package(archive, output)
    assert (output / "flowers-resnet9.pth").read_bytes() == b"model"
    assert (output / "classes.json").read_bytes() == b'["daisy"]'


def test_bad_artifact_does_not_replace_existing_files(tmp_path, monkeypatch):
    archive = package_fixture(tmp_path, monkeypatch, expected_weight=b"different")
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "flowers-resnet9.pth").write_bytes(b"previous")
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        fetch_model.install_package(archive, output)
    assert (output / "flowers-resnet9.pth").read_bytes() == b"previous"
    assert not (output / "classes.json").exists()


def test_bad_package_rejected_before_extracting(tmp_path, monkeypatch):
    archive = package_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(fetch_model, "PACKAGE_SHA256", "incorrect")
    with pytest.raises(ValueError, match="Package checksum"):
        fetch_model.install_package(archive, tmp_path / "out")


def test_llm_kill_switch_wins_over_existing_credentials(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "true")
    for name in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
        monkeypatch.setenv(name, "dummy-test-value")
    assert make_chat_model() is None


def test_phone_orientation_survives_memory_optimized_decode():
    image = Image.new("RGB", (3, 2), "yellow")
    metadata = image.getexif()
    metadata[274] = 6
    data = BytesIO()
    image.save(data, "JPEG", exif=metadata)
    decoded = decode_image(data.getvalue())
    assert decoded.size == (2, 3)
    assert decoded.mode == "RGB"
    decoded.close()


def test_predictions_do_not_accumulate_decoded_images():
    class RecordingPredictor:
        ready, error, classes = True, None, ["daisy"]
        active, peak = 0, 0
        lock = threading.Lock()

        def load(self):
            pass

        def predict(self, image):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            image.getpixel((0, 0))
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return {"predicted_class": "daisy"}

    predictor = RecordingPredictor()
    buffer = BytesIO()
    Image.new("RGB", (10, 10), "white").save(buffer, "PNG")
    with TestClient(create_app(predictor)) as client:
        def predict(_):
            return client.post("/predict", files={"file": ("sample.png", buffer.getvalue())}).status_code
        with ThreadPoolExecutor(max_workers=4) as executor:
            assert list(executor.map(predict, range(4))) == [200] * 4
    assert predictor.peak == 1
