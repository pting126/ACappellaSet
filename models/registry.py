"""Dependency-free model discovery, including explicit publication status."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries():
    return json.loads((ROOT / "manifests/models.json").read_text())["models"]


def resolve(name):
    for entry in entries():
        if name in [entry["id"], *entry.get("aliases", [])]:
            return entry
    raise ValueError(f"Unknown model {name!r}; use --list-models to see supported IDs.")


def print_models():
    print("Model\tFamily\tOutputs\tCheckpoint download")
    for entry in entries():
        available = bool(entry.get("portable_checkpoint", {}).get("download_url"))
        print("\t".join([entry["id"], entry["family"], ",".join(entry["source_order"]),
                         "available" if available else "pending; local checkpoint only"]))
