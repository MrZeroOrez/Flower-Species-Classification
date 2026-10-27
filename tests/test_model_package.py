import json
from zipfile import ZipFile

import pytest

from scripts.package_model import package_model, sha256


@pytest.fixture
def evaluated(tmp_path):
    (tmp_path / "flowers-resnet9.pth").write_bytes(b"evaluated checkpoint")
    (tmp_path / "classes.json").write_text('["rose", "tulip"]')
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}')
    metrics = {"checkpoint_sha256": sha256(tmp_path / "flowers-resnet9.pth"),
               "manifest_sha256": sha256(manifest), "classes": ["rose", "tulip"]}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path, manifest, tmp_path / "release.zip"


def test_package_contains_evaluation_but_no_images_and_cannot_be_overwritten(evaluated):
    model_dir, manifest, output = evaluated
    result = package_model(model_dir, manifest, output)
    assert result["package_sha256"] == sha256(output)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"artifacts/flowers-resnet9.pth", "artifacts/classes.json",
                                           "artifacts/metrics.json", "artifacts/dataset_manifest.json",
                                           "MODEL_README.txt"}
    with pytest.raises(FileExistsError):
        package_model(model_dir, manifest, output)


@pytest.mark.parametrize("filename,content", [("flowers-resnet9.pth", "different weights"),
                                               ("manifest.json", "different split"),
                                               ("classes.json", '["tulip", "rose"]')])
def test_package_rejects_mixed_training_artifacts(evaluated, filename, content):
    model_dir, manifest, output = evaluated
    (model_dir / filename).write_text(content)
    with pytest.raises(ValueError):
        package_model(model_dir, manifest, output)
    assert not output.exists()
