from io import BytesIO
from zipfile import ZipFile

from PIL import Image
import pytest

from scripts.prepare_data import CLASSES, prepare


def test_reproducible_splits_exclude_duplicates_and_invalid_images(tmp_path):
    archive = tmp_path / "flowers-recognition.zip"
    with ZipFile(archive, "w") as bundle:
        first = None
        for index, label in enumerate(CLASSES):
            for item in range(12):
                buffer = BytesIO()
                Image.new("RGB", (8, 8), (index * 40, item * 15, 100)).save(buffer, "PNG")
                data = buffer.getvalue()
                bundle.writestr(f"flowers/{label}/{item}.png", data)
                if first is None:
                    first = data
        bundle.writestr("flowers/daisy/duplicate.png", first)
        bundle.writestr("flowers/daisy/broken.jpg", b"invalid")
    result = prepare(tmp_path)
    assert len(result["records"]) == 60
    assert len(result["invalid_files"]) == 1
    assert len(result["duplicate_files_excluded"]) == 1
    assert len({r["pixel_sha256"] for r in result["records"]}) == 60
    assert result["records"] == prepare(tmp_path)["records"]
    assert all(set(counts) == set(CLASSES) for counts in result["split_counts"].values())


def test_conflicting_labels_are_excluded(tmp_path):
    archive = tmp_path / "flowers-recognition.zip"
    with ZipFile(archive, "w") as bundle:
        for index, label in enumerate(CLASSES):
            for item in range(12):
                buffer = BytesIO()
                Image.new("RGB", (8, 8), (index * 40, item * 15, 100)).save(buffer, "PNG")
                bundle.writestr(f"flowers/{label}/{item}.png", buffer.getvalue())
        buffer = BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, "PNG")
        for label in ["rose", "tulip"]:
            bundle.writestr(f"flowers/{label}/conflict.png", buffer.getvalue())
    result = prepare(tmp_path)
    assert len(result["ambiguous_files_excluded"]) == 2
    assert len(result["records"]) == 60


def test_reject_archive_path_traversal(tmp_path):
    with ZipFile(tmp_path / "flowers-recognition.zip", "w") as bundle:
        bundle.writestr("../escape.jpg", b"not allowed")
    with pytest.raises(ValueError, match="Unexpected archive entry"):
        prepare(tmp_path)
    assert not (tmp_path.parent / "escape.jpg").exists()
