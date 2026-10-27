"""Fetch the separately delivered baseline package at build time; verify provenance.

Render: set MODEL_PACKAGE_URL to a public HTTPS download (e.g. a GitHub release
asset). Offline verification: python -m scripts.fetch_model --archive PATH.
Only the two required artifacts are extracted, never arbitrary archive paths.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

PACKAGE_SHA256 = "e527ebfd8f64a7098f3999cf09ce952371e01ad2ef39551cd06d049a27f710ac"
ARTIFACTS = {
    "flowers-resnet9.pth": "f27aa366cf3700d20ecd54bf5e40e2af995d10cf5c7ee3358f0a4cfdda493eb8",
    "classes.json": "01118f4ea759247969a766910c0864f1a186ccfd161b5285d719692906618277",
}
MAX_PACKAGE_BYTES = 40 * 1024 * 1024


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def install_package(archive, output):
    archive, output = Path(archive), Path(output)
    if archive.stat().st_size > MAX_PACKAGE_BYTES or digest(archive) != PACKAGE_SHA256:
        raise ValueError("Package checksum/size mismatch. Use the exact delivered baseline ZIP.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary)
        with ZipFile(archive) as package:
            for name, expected in ARTIFACTS.items():
                entry = package.getinfo("artifacts/" + name)
                if entry.file_size > MAX_PACKAGE_BYTES:
                    raise ValueError("Artifact exceeds size limit")
                with package.open(entry) as source, (staged / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                if digest(staged / name) != expected:
                    raise ValueError("Artifact checksum mismatch: " + name)
        # Both artifacts are verified before replacing either existing file.
        for name in ARTIFACTS:
            (staged / name).replace(output / name)


def fetch(url, output):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("MODEL_PACKAGE_URL must be a public HTTPS URL without embedded credentials")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "model.zip"
        with urlopen(url, timeout=120) as response, archive.open("wb") as target:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Model download redirected away from HTTPS")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_PACKAGE_BYTES:
                    raise ValueError("Model package exceeds size limit")
                target.write(chunk)
        install_package(archive, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.archive:
        install_package(args.archive, args.output)
    elif all((args.output / name).is_file() and digest(args.output / name) == expected
             for name, expected in ARTIFACTS.items()):
        print("Verified existing baseline artifacts")
    else:
        package_url = os.getenv("MODEL_PACKAGE_URL", "")
        if not package_url:
            raise SystemExit("MODEL_PACKAGE_URL is required. Publish the provided model ZIP as a GitHub release asset and supply its public download URL. See docs/DEPLOY_RENDER.md.")
        fetch(package_url, args.output)
    print("Baseline model artifacts verified")
