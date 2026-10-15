"""Dependency-free model discovery, including explicit publication status."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries():
    fixed = json.loads((ROOT / "manifests/models.json").read_text())["models"]
    conditioned = json.loads((ROOT / "manifests/conditioned.json").read_text())["models"]
    return fixed + conditioned


def resolve(name):
    for entry in entries():
        if name in [entry["id"], *entry.get("aliases", [])]:
            return entry
    raise ValueError(f"Unknown model {name!r}; use --list-models to see supported IDs.")


def print_models():
    print("Model\tFamily\tOutputs\tCheckpoint download\tSelections")
    for entry in entries():
        available = bool(entry.get("portable_checkpoint", {}).get("download_url"))
        print("\t".join([entry["id"], entry["family"], ",".join(entry["source_order"]),
                         "available" if available else "pending; local checkpoint only",
                         ",".join(label for selection in entry.get('selections', [])
                                   for label in selection['selection_labels'])]))


def select_checkpoint(spec, selection=None):
    """Resolve a validation criterion without choosing from test performance."""
    candidates = spec.get('selections')
    if candidates is None:
        if selection is not None:
            raise ValueError('--selection applies only to registered conditioned selections')
        return spec
    if selection is None:
        if len(candidates) != 1:
            raise ValueError('This model requires --selection; use --list-models for criteria')
        chosen = candidates[0]
    else:
        matches = [item for item in candidates if selection in item['selection_labels']]
        if len(matches) != 1:
            raise ValueError(f'No unique released checkpoint for selection {selection!r}')
        chosen = matches[0]
    return dict(spec, **chosen, selection=selection or chosen['selection_labels'][0])
