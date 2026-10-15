"""Download a registered portable model anonymously and verify its SHA-256."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Fixed_6stem_htdemucs_SKnJC")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Destination directory")
    args = parser.parse_args()
    registry = json.loads((ROOT / "manifests/models.json").read_text())
    models = {name: model for model in registry["models"]
              for name in [model["id"], *model.get("aliases", [])]}
    if args.model not in models:
        parser.error(f"Unknown model {args.model!r}. Available: {', '.join(models)}")
    artifact = models[args.model].get("portable_checkpoint", {})
    url = artifact.get("download_url")
    if not url:
        parser.error("This model has no published portable checkpoint yet.")
    if not url.startswith("https://"):
        parser.error("Registered downloads must use HTTPS.")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / artifact["filename"]
    if destination.exists():
        if sha256(destination) != artifact["sha256"]:
            raise ValueError(f"Existing file has an unexpected hash: {destination}")
        print(f"Already verified: {destination}")
        return
    # Publish only fully verified bytes; never replace a concurrently created file.
    with tempfile.NamedTemporaryFile(prefix=".download-", dir=args.output) as temporary:
        with urllib.request.urlopen(url, timeout=60) as response:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                temporary.write(block)
        temporary.flush()
        path = Path(temporary.name)
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError("Downloaded checkpoint does not match the registered size and SHA-256.")
        os.link(path, destination)
    print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
