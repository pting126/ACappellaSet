"""Offline, hash-verified bridge from the v1.0 release to historical audio roots."""

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STEMS = ('soprano', 'alto', 'tenor', 'baritone', 'bass', 'vp')
BLOCK = 262144
DOCUMENTS = ('file_manifest.csv', 'metadata/metadata.csv', 'metadata/splits.csv')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Unsafe release path: ' + value)
    return Path(path)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def historical_references():
    """Keep existing scientific manifests authoritative, including their hashes."""
    index = json.loads((ROOT / 'manifests/splits/index.json').read_text())
    references = {}
    for entry in index['sets'].values():
        for filename, info in entry['files'].items():
            if not filename.endswith('.csv'):
                continue
            raw = (ROOT / info['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise ValueError('Historical manifest checksum mismatch: ' + filename)
            decoded = gzip.decompress(raw)
            if hashlib.sha256(decoded).hexdigest() != info['content_sha256']:
                raise ValueError('Historical manifest content checksum mismatch: ' + filename)
            for row in csv.DictReader(io.StringIO(decoded.decode())):
                for field in ('mixture_path', 'target_path', 'enroll_path'):
                    value = row.get(field, '')
                    if value.split('/')[0] not in ('SK', 'AB'):
                        continue
                    relative_path(value)
                    expected = row.get('source_audio_sha256') if (
                        field == 'target_path' or
                        (field == 'enroll_path' and filename == 'enrollment_index.csv')) else None
                    previous = references.get(value)
                    if previous and expected and previous != expected:
                        raise ValueError('Conflicting historical hashes: ' + value)
                    references[value] = expected or previous
    return references


def plan_release(source):
    """Validate pinned metadata and identities before reading or generating audio."""
    source = Path(source).resolve()
    registry = json.loads((ROOT / 'manifests/dataset.json').read_text())
    registered = {entry['path']: entry for entry in registry['files']}
    documents = {}
    for name in DOCUMENTS:
        path = source / name
        if not path.is_file():
            raise ValueError('Missing release file: ' + str(path))
        actual = digest(path)
        if actual != registered[name]['sha256']:
            raise ValueError('Release metadata checksum mismatch: ' + name +
                             '; use the dataset revision registered in manifests/dataset.json')
        documents[name] = actual
    splits = {}
    for row in read_csv(source / 'metadata/splits.csv'):
        if row['song_id'] in splits:
            raise ValueError('Duplicate song split: ' + row['song_id'])
        splits[row['song_id']] = row['split']
    inventory = json.loads((ROOT / 'manifests/splits/track_inventory.json').read_text())
    sk_splits = {}
    for split, tracks in inventory['corpora']['SK'].items():
        for track in tracks:
            if track in sk_splits:
                raise ValueError('Historical track in multiple splits: ' + track)
            sk_splits[track] = split
    recordings = {}
    used_songs = set()
    for row in read_csv(source / 'metadata/metadata.csv'):
        song = row['song_id']
        spatial = {'stereo_spatial': 'SP', 'dual_mono': 'DM'}[row['spatial_type']]
        track = song + '_' + row['variant'] + '_' + spatial
        relative_path(track)
        if '/' in track or track in recordings:
            raise ValueError('Invalid or duplicate recording identity: ' + track)
        if row['Group'] == 'Seagull-K':
            split = sk_splits.get(track)
            if split is None or splits.get(song) != split.removesuffix('_flat'):
                raise ValueError('Release/historical split mismatch: ' + track)
            dest = Path('SK') / split / track
        elif row['Group'] in ('Annappella', 'REAL A Cappella'):
            if splits.get(song) != 'external_test':
                raise ValueError('AB must remain external_test: ' + track)
            dest = Path('AB') / track
        else:
            raise ValueError('Unknown vocal group: ' + row['Group'])
        if row['is_primary'] not in ('0', '1'):
            raise ValueError('Invalid primary flag: ' + track)
        folder = Path('audio/primary') / song if row['is_primary'] == '1' else (
            Path('audio/alternates') / song / row['variant'])
        roles = [role for role in STEMS if row[role] != '0']
        if not all(row[role].strip() for role in STEMS) or int(row['stem_count']) != len(roles):
            raise ValueError('Invalid active-role metadata: ' + track)
        recordings[track] = {'track': track, 'destination': dest.as_posix(),
                             'folder': folder.as_posix(), 'roles': roles, 'files': {}}
        used_songs.add(song)
    if used_songs != set(splits) or set(sk_splits) != {
            k for k, v in recordings.items() if v['destination'].startswith('SK/')}:
        raise ValueError('Release recording membership differs from historical inventory')
    paths = set()
    for row in read_csv(source / 'file_manifest.csv'):
        track, role = row['recording_id'], row['role']
        if track not in recordings or role not in (*STEMS, 'mixture'):
            raise ValueError('Unknown recording or role in file manifest')
        rec = recordings[track]
        path = relative_path(row['path'])
        if path != Path(rec['folder']) / (role + '.wav') or row['path'] in paths or role in rec['files']:
            raise ValueError('Duplicate or mismatched audio path: ' + row['path'])
        paths.add(row['path'])
        if (int(row['sample_rate']), int(row['channels'])) != (44100, 2) or int(row['frames']) < 1:
            raise ValueError('Invalid registered audio dimensions: ' + row['path'])
        rec['files'][role] = row
    mapped = {}
    for rec in recordings.values():
        if set(rec['files']) != set(rec['roles']) | {'mixture'}:
            raise ValueError('Audio files differ from annotated active roles: ' + rec['track'])
        frames = {int(f['frames']) for f in rec['files'].values()}
        if len(frames) != 1:
            raise ValueError('Misaligned recording: ' + rec['track'])
        rec['frames'] = frames.pop()
        rec['missing_roles'] = [role for role in STEMS if role not in rec['roles']]
        for role in (*STEMS, 'mixture'):
            mapped[rec['destination'] + '/' + role + '.wav'] = rec['files'].get(role)
    historical = historical_references()
    for path, expected in historical.items():
        if path not in mapped:
            raise ValueError('Historical input missing from release mapping: ' + path)
        if expected and (mapped[path] is None or mapped[path]['sha256'] != expected):
            raise ValueError('Historical source hash differs from release: ' + path)
    return {'dataset_revision': registry['revision'], 'metadata_sha256': documents,
            'historical_referenced_files': len(historical),
            'historical_hashes_matched': sum(bool(x) for x in historical.values()),
            'recordings': [recordings[k] for k in sorted(recordings)]}


def verify_audio(path, entry):
    import numpy as np
    import soundfile as sf
    if path.stat().st_size != int(entry['bytes']) or digest(path) != entry['sha256']:
        raise ValueError('Release audio checksum/size mismatch: ' + str(path))
    with sf.SoundFile(path) as reader:
        actual = (reader.samplerate, reader.channels, reader.frames, reader.subtype)
        expected = (int(entry['sample_rate']), int(entry['channels']),
                    int(entry['frames']), entry['subtype'])
        if actual != expected:
            raise ValueError('Audio header differs from release manifest: ' + str(path))
        for block in reader.blocks(BLOCK, dtype='float32', always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError('Non-finite release audio: ' + str(path))


def write_silence(path, frames):
    import numpy as np
    import soundfile as sf
    zero = np.zeros((min(BLOCK, frames), 2), dtype=np.float32)
    with sf.SoundFile(path, 'w', samplerate=44100, channels=2, subtype='PCM_16') as writer:
        for start in range(0, frames, BLOCK):
            writer.write(zero[:min(BLOCK, frames - start)])


def prepare_release(source, output, link_mode='symlink'):
    """Preserve released bytes, pad explicitly absent roles, never download audio."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if link_mode not in ('symlink', 'copy'):
        raise ValueError('Use symlink or copy')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output must be a new directory: ' + str(output))
    # Resolve parent aliases (e.g. macOS /var -> /private/var) before making relative links.
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError('Output must be outside the release input tree')
    plan = plan_release(source)
    for rec in plan['recordings']:
        for entry in rec['files'].values():
            if not (source / entry['path']).is_file():
                raise ValueError('Missing extracted audio: ' + str(source / entry['path']) +
                                 '; extract both audio ZIPs into the release root first')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {**plan, 'source_root': str(source), 'output_root': str(output),
               'link_mode': link_mode, 'sample_rate': 44100, 'channels': 2,
               'mixtures': 'Released mixtures preserved byte-for-byte; no summation or normalization',
               'padding': 'PCM_16 digital silence for metadata roles marked 0 only',
               'protocol': 'Historical scientific manifests unchanged; PC01 target exclusion retained',
               'roots': {corpus: str(output / corpus) for corpus in ('SK', 'AB')},
               'outputs': {}}
    with tempfile.TemporaryDirectory(prefix='.release-inputs-', dir=output.parent) as temp:
        staging = Path(temp) / 'result'
        staging.mkdir()
        for rec in plan['recordings']:
            dest = staging / rec['destination']
            dest.mkdir(parents=True)
            for role, entry in rec['files'].items():
                src = source / entry['path']
                verify_audio(src, entry)
                rel = Path(rec['destination']) / (role + '.wav')
                target = staging / rel
                if link_mode == 'symlink':
                    target.symlink_to(os.path.relpath(src, (output / rel).parent))
                else:
                    shutil.copyfile(src, target)
                    if digest(target) != entry['sha256']:
                        raise ValueError('Copied file checksum mismatch: ' + str(src))
                receipt['outputs'][rel.as_posix()] = {'kind': link_mode,
                    'source': entry['path'], 'sha256': entry['sha256'], 'frames': rec['frames']}
            for role in rec['missing_roles']:
                rel = Path(rec['destination']) / (role + '.wav')
                write_silence(staging / rel, rec['frames'])
                receipt['outputs'][rel.as_posix()] = {'kind': 'generated_silence',
                    'sha256': digest(staging / rel), 'frames': rec['frames']}
        # Refuse to publish a layout if source audio or metadata changed during construction.
        for rec in plan['recordings']:
            for entry in rec['files'].values():
                if digest(source / entry['path']) != entry['sha256']:
                    raise ValueError('Source audio changed during preparation: ' + entry['path'])
        for name, expected in plan['metadata_sha256'].items():
            if digest(source / name) != expected:
                raise ValueError('Source metadata changed during preparation: ' + name)
        (staging / 'release_preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists() or output.is_symlink():
            raise FileExistsError('Output appeared during preparation: ' + str(output))
        staging.rename(output)
    return receipt
