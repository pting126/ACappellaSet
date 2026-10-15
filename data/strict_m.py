"""Opt-in fail-closed loader; no changes to historical V2-A datasets."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from .present import V2APresentDataset, keyed_seed
from .residual_policy import require_usable_references
from .policy_m import legal_enrollment as legal_recording_enrollment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


class StrictPresentDataset(V2APresentDataset):
    def __init__(self, manifest_dir, split, conditioning, **kwargs):
        root = Path(manifest_dir)
        policy = json.loads((root / 'DATA_PROTOCOL.json').read_text())
        complete = json.loads((root / 'COMPLETE.json').read_text())
        if complete['protocol_sha256'] != digest(root / 'DATA_PROTOCOL.json'):
            raise ValueError('Manifest protocol changed')
        if policy['version'] != 'm_combined_klmatched_v1' or policy['status'] != 'DATA_GO_PENDING_GPU_PREFLIGHT':
            raise ValueError('Dataset not certified for M preflight')
        if split != 'train' or conditioning not in policy['permitted_conditioning']:
            raise ValueError('Use the separate frozen full-song validation portal')
        for name, field in [('target_examples.csv', 'target_sha256'), ('enrollment_index.csv', 'enrollment_sha256')]:
            if digest(root / name) != policy[field]:
                raise ValueError('Manifest changed: ' + name)
        if kwargs.get('target_seconds', 10) != 10 or kwargs.get('enroll_seconds', 10) != 10:
            raise ValueError('M requires 10s target/enrollment')
        if kwargs.get('target_parts'):
            raise ValueError('Hidden per-model target population filtering prohibited')
        # Base role initialization reads the identical target rows without an
        # O(target rows x gallery rows) eligibility scan. No audio is decoded.
        super().__init__(root, split, 'role', **kwargs)
        self.conditioning = conditioning
        self.policy = policy
        self.strict_pools = {}
        if any(r['part'] not in policy['training_roles'] or '+' in r['singer_id']
               or r['singer_id'] in ('', '0', 'None') or r.get('is_dummy', '0') != '0'
               or r.get('exclusion_reason', '') for r in self.rows):
            raise ValueError('Invalid positive-query identity/role')
        if kwargs.get('max_examples') is None and len(self.rows) != policy['target_count']:
            raise ValueError('Target population count changed')
        if conditioning in ('enrollment', 'dual'):
            by_singer = defaultdict(list)
            with (root / 'enrollment_index.csv').open(newline='') as f:
                for r in csv.DictReader(f):
                    if r['split'] != 'train' or float(r['enroll_end'])-float(r['enroll_start']) != 10:
                        raise ValueError('Illegal training gallery candidate')
                    by_singer[r['singer_id']].append(r)
            unique_targets = {(r['singer_id'], r['song_id'], r['part'], r['source_audio_sha256']): r for r in self.rows}
            for (singer, composition, part, digest_value), target in unique_targets.items():
                pool = [r for r in by_singer[singer] if legal_recording_enrollment(target, r)]
                if not pool:
                    raise ValueError('No different-composition enrollment: ' + singer + '/' + composition)
                self.strict_pools[singer, composition, part, digest_value] = pool

    def _choose_enrollment(self, target, index):
        pool = self.strict_pools[target['singer_id'], target['song_id'], target['part'], target['source_audio_sha256']]
        rng = random.Random(keyed_seed(self.seed, self.epoch, index, 'enrollment'))
        candidate = pool[rng.randrange(len(pool))]
        if (not legal_recording_enrollment(target, candidate)
                or candidate['enroll_path'] == target['target_path']):
            raise ValueError('Strict query identity/composition violation')
        return candidate

    def __getitem__(self, index):
        result = super().__getitem__(index)
        for key in ('mixture', 'target', 'enrollment'):
            audio = result[key]
            if tuple(audio.shape) != (2, 441000) or not torch.isfinite(audio).all():
                raise ValueError('Invalid runtime waveform: ' + key)
        require_usable_references(result['mixture'], result['target'])
        if self.conditioning in ('enrollment', 'dual'):
            if not result['enrollment'].mean(dim=0).float().var(unbiased=False) > 0:
                raise ValueError('Silent mono enrollment')
        elif torch.count_nonzero(result['enrollment']):
            raise ValueError('Role-only enrollment must remain disabled')
        return result
