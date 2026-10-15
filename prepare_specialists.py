"""Prepare original mixture-only specialist inputs from pre-split six-stem audio."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

# Summation order is part of the original float32 preprocessing recipe.
STEMS = {"JC": ("soprano", "alto", "tenor", "bass", "vp", "lead_vocal"),
         "SK": ("soprano", "alto", "tenor", "baritone", "bass", "vp")}
SPLITS = {"JC": ("train", "valid", "test"), "SK": ("train_flat", "valid_flat", "test_flat")}
COUNTS = {"JC": (30, 10, 10), "SK": (69, 11, 20)}
BLOCK = 262144


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    info = sf.info(path)
    if info.frames <= 0 or (info.samplerate, info.channels) != (44100, 2):
        raise ValueError(f"Expected nonempty 44.1 kHz stereo audio: {path}")
    return info.frames, info.samplerate, info.channels


def write_residual(paths, destination, subtract=False):
    frames, rate, channels = signature(paths[0])
    if any(signature(p) != (frames, rate, channels) for p in paths):
        raise ValueError("All mixture and stem files must have identical audio dimensions")
    with ExitStack() as stack:
        readers = [stack.enter_context(sf.SoundFile(p)) for p in paths]
        writer = stack.enter_context(sf.SoundFile(destination, 'w', samplerate=rate,
                                                  channels=channels, subtype='FLOAT'))
        for start in range(0, frames, BLOCK):
            n = min(BLOCK, frames - start)
            total = np.zeros((n, channels), dtype=np.float32)
            for index, reader in enumerate(readers):
                audio = reader.read(n, dtype='float32', always_2d=True)
                if audio.shape != total.shape or not np.isfinite(audio).all():
                    raise ValueError(f"Invalid audio in {reader.name}")
                if subtract and index == 1:
                    total -= audio
                else:
                    total += audio
            if not np.isfinite(total).all():
                raise ValueError("Non-finite derived samples")
            writer.write(total)


def prepare(corpus, source, output, targets=None, allow_subset=False, historical_role_filter=False):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the input tree")
    stems = STEMS[corpus]
    inventory = json.loads((Path(__file__).parent/'manifests/splits/track_inventory.json').read_text())
    if historical_role_filter and (corpus != 'SK' or allow_subset):
        raise ValueError('Historical role filtering requires the complete SK input inventory')
    filters = inventory['specialist_filters'].get(corpus, {}) if historical_role_filter else {}
    targets = list(stems if targets is None else targets)
    if not targets or len(set(targets)) != len(targets) or any(t not in stems for t in targets):
        raise ValueError(f"Targets must be distinct members of {stems}")
    tracks = []
    identities = set()
    for split, count in zip(SPLITS[corpus], COUNTS[corpus]):
        split_root = source / split
        entries = sorted(p for p in split_root.iterdir() if p.is_dir())
        if not allow_subset and len(entries) != count:
            raise ValueError(f"{split}: expected {count} tracks, found {len(entries)}")
        if not allow_subset and [p.name for p in entries] != inventory['corpora'][corpus][split]:
            raise ValueError(f'{split}: historical track membership changed')
        for track in entries:
            if track.name in identities:
                raise ValueError(f"Track appears in multiple splits: {track.name}")
            identities.add(track.name)
            paths = [track / (stem + '.wav') for stem in stems]
            if corpus == 'JC':
                paths.append(track / 'mixture.wav')
            signatures = [signature(p) for p in paths]
            if len(set(signatures)) != 1:
                raise ValueError(f"Misaligned files: {track}")
            tracks.append((split, track, paths))
    if not tracks:
        raise ValueError("No input tracks")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as scratch:
        staging = Path(scratch) / 'result'
        staging.mkdir()
        receipt = {'corpus': corpus, 'targets': targets, 'historical_counts_enforced': not allow_subset,
                   'historical_role_filter': historical_role_filter,
                   'source_root': str(source), 'sample_rate': 44100, 'channels': 2,
                   'recipe': 'sum_other_stems' if corpus == 'JC' else 'additive_mixture_minus_target',
                   'summation_order': list(stems), 'gain': 1.0, 'tracks': []}
        for split, track, paths in tracks:
            before = {p.name: sha256(p) for p in paths}
            for path in paths:
                with sf.SoundFile(path) as reader:
                    for block in reader.blocks(BLOCK, dtype='float32', always_2d=True):
                        if not np.isfinite(block).all():
                            raise ValueError(f"Non-finite input: {path}")
            shared = staging / 'mixtures' / split / track.name
            shared.mkdir(parents=True)
            mixture = shared / 'mixture.wav'
            if corpus == 'SK':
                write_residual([track / (s + '.wav') for s in stems], mixture)
            else:
                # Preserve the recorded JC mixture; do not silently reconstruct it.
                shutil.copyfile(track / 'mixture.wav', mixture)
                with sf.SoundFile(mixture) as reader:
                    for block in reader.blocks(BLOCK, dtype='float32', always_2d=True):
                        if not np.isfinite(block).all():
                            raise ValueError(f"Non-finite mixture: {track}")
            outputs = {'mixture': sha256(mixture)}
            for target in targets:
                if target in filters and track.name not in filters[target][split]:
                    continue
                dest = staging / target / split / track.name
                dest.mkdir(parents=True)
                # All symlinks stay inside this output tree and survive moving it.
                (dest / 'mixture.wav').symlink_to(Path('../../../mixtures') / split / track.name / 'mixture.wav')
                target_path = dest / (target + '.wav')
                shutil.copyfile(track / (target + '.wav'), target_path)
                residual = dest / ('non' + target + '.wav')
                inputs = [mixture, target_path] if corpus == 'SK' else [track / (s + '.wav') for s in stems if s != target]
                write_residual(inputs, residual, subtract=corpus == 'SK')
                outputs[target] = {'target_sha256': sha256(target_path), 'residual_sha256': sha256(residual)}
            if before != {p.name: sha256(p) for p in paths}:
                raise ValueError(f"Source files changed during preparation: {track}")
            receipt['tracks'].append({'split': split, 'track_id': track.name,
                                      'input_sha256': before, 'outputs': outputs})
        (staging / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', choices=STEMS, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Existing pre-split six-stem root')
    parser.add_argument('--output', type=Path, required=True, help='New destination directory')
    parser.add_argument('--targets', nargs='+', help='Native stem filenames without .wav; default: all six')
    parser.add_argument('--allow-subset', action='store_true', help='Skip historical track counts for custom data or smoke tests')
    parser.add_argument('--historical-role-filter', action='store_true',
                        help='Use the exact original SK Soprano/Baritone real-target inventories')
    args = parser.parse_args()
    receipt = prepare(args.corpus, args.input, args.output, args.targets, args.allow_subset,
                      args.historical_role_filter)
    print(f"Prepared {len(receipt['tracks'])} tracks for {len(receipt['targets'])} specialists in {args.output}")


if __name__ == '__main__':
    main()
