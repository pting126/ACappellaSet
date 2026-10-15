"""Explicit evaluation scopes; each retained track/role pair has weight one."""

from collections import defaultdict
import math

SHARED = frozenset(('S', 'A', 'T', 'B', 'VP'))
NATIVE = {'JC': SHARED | {'Lead'}, 'SK': SHARED | {'Bar'}, 'AB': SHARED | {'Bar'}}
PROTOCOLS = ('mixture_only', 'part_query', 'strict_enrollment', 'fallback_inclusive')
IDENTITY = ('model', 'corpus', 'protocol', 'song_id', 'track_id', 'role')


def validate_identity(row):
    if any(not row.get(key, '').strip() for key in IDENTITY):
        raise ValueError(f"Missing required identity fields: {IDENTITY}")
    if row['corpus'] not in NATIVE or row['role'] not in NATIVE[row['corpus']]:
        raise ValueError(f"Unsupported corpus/role: {row['corpus']}/{row['role']}")
    if row['protocol'] not in PROTOCOLS:
        raise ValueError(f"Protocol must be one of {PROTOCOLS}")
    # An obvious track/composition disagreement must not bypass the exclusion.
    if row['corpus'] == 'AB' and (row['track_id'] == 'PC01' or row['track_id'].startswith('PC01_')):
        if row['song_id'] != 'PC01':
            raise ValueError('PC01 track conflicts with authoritative song_id')


def summarize(rows):
    groups = defaultdict(list)
    seen = set()
    track_songs = {}
    for row in rows:
        validate_identity(row)
        key = tuple(row[k] for k in ('model', 'corpus', 'protocol', 'track_id', 'role'))
        if key in seen:
            raise ValueError(f'Duplicate evaluated pair: {key}')
        seen.add(key)
        track_key = row['corpus'], row['track_id']
        if track_songs.setdefault(track_key, row['song_id']) != row['song_id']:
            raise ValueError('Conflicting song_id for the same track')
        if row['corpus'] == 'AB' and row['song_id'] == 'PC01':
            raise ValueError('AB PC01 must be excluded before aggregation')
        if not all(math.isfinite(float(row[k])) for k in ('si_sdr', 'mix_si_sdr', 'si_sdri')):
            raise ValueError('Non-finite scores')
        if not math.isclose(float(row['si_sdr']) - float(row['mix_si_sdr']), float(row['si_sdri']), abs_tol=1e-9, rel_tol=0):
            raise ValueError('SI-SDRi decomposition failed')
        groups[tuple(row[k] for k in ('model', 'corpus', 'protocol'))].append(row)
    summaries = []
    for (model, corpus, protocol), items in sorted(groups.items()):
        for scope, roles in [('shared', SHARED), ('inclusive', NATIVE[corpus])]:
            retained = [r for r in items if r['role'] in roles]
            summaries.append({'model': model, 'corpus': corpus, 'protocol': protocol, 'scope': scope,
                              'n_pairs': len(retained), 'n_tracks': len({r['track_id'] for r in retained}),
                              'roles_present': sorted({r['role'] for r in retained}),
                              'overall_si_sdri': sum(float(r['si_sdri']) for r in retained) / len(retained) if retained else None,
                              'per_role': {role: {'n_pairs': sum(r['role'] == role for r in retained),
                                                  'si_sdri': sum(float(r['si_sdri']) for r in retained if r['role'] == role) / sum(r['role'] == role for r in retained)}
                                           for role in sorted({r['role'] for r in retained})}})
    return summaries
