"""Resolve a safe inference export produced by this repository's trainer."""
import json
from pathlib import Path

from .registry import resolve


def resolve_export(manifest, checkpoint, selection=None):
    if selection is not None:
        raise ValueError('A new training export has no historical paper selection')
    manifest=Path(manifest)
    record=json.loads(manifest.read_text())
    if record.get('format')!='acappellaset-conditioned-training-v1':
        raise ValueError('Unknown training export format')
    spec=resolve(record['model'])
    if spec['family']!='conditioned-htdemucs' or record['config_sha256']!=spec['config_sha256']:
        raise ValueError('Training export recipe/config mismatch')
    if Path(record['file']).name!=record['file'] or Path(checkpoint).resolve()!=(manifest.parent/record['file']).resolve():
        raise ValueError('Checkpoint must be the adjacent file named by the training export')
    if not isinstance(record['step'],int) or record['step']<1:
        raise ValueError('Invalid training step')
    return dict(spec,checkpoint_sha256=record['sha256'],step=record['step'],
                selection='new_training_run',safe_training_export=True)
