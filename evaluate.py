"""Score explicitly paired full-song WAVs; no inference or checkpoint selection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from evaluation.reporting import IDENTITY, summarize, validate_identity


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate(manifest, output):
    import numpy as np
    import soundfile as sf
    import torch
    from evaluation.metrics import dbfs, si_sdr_value

    manifest, output = Path(manifest).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be new: {output}')
    torch.set_num_threads(2)
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty evaluation manifest')
    results, excluded, seen, baselines, track_songs = [], [], set(), {}, {}
    for row in rows:
        validate_identity(row)
        identity = {k: row[k] for k in IDENTITY}
        key = tuple(row[k] for k in ('model', 'corpus', 'protocol', 'track_id', 'role'))
        if key in seen:
            raise ValueError(f'Duplicate manifest pair: {key}')
        seen.add(key)
        track_key = row['corpus'], row['track_id']
        if track_songs.setdefault(track_key, row['song_id']) != row['song_id']:
            raise ValueError('Conflicting song_id for the same track')
        if row['corpus'] == 'AB' and row['song_id'] == 'PC01':
            excluded.append(dict(identity, reason='AB excluding PC01'))
            continue
        audio, hashes = {}, {}
        for kind in ('mixture', 'reference', 'estimate'):
            if not row.get(kind):
                raise ValueError(f'Missing {kind} path for {key}')
            path = manifest.parent / row[kind]
            before = sha256(path)
            samples, rate = sf.read(path, dtype='float32', always_2d=True)
            if rate != 44100 or samples.shape[1] != 2 or not len(samples) or not np.isfinite(samples).all():
                raise ValueError(f'Expected finite, nonempty stereo 44.1 kHz audio: {path}')
            if sha256(path) != before:
                raise ValueError(f'Audio changed while reading: {path}')
            audio[kind] = torch.from_numpy(samples.T.copy())
            hashes[kind + '_sha256'] = before
        if len({tuple(a.shape) for a in audio.values()}) != 1:
            raise ValueError(f'Audio lengths must match exactly: {key}')
        baseline_key = row['corpus'], row['track_id'], row['role']
        signature = hashes['mixture_sha256'], hashes['reference_sha256']
        if baselines.setdefault(baseline_key, signature) != signature:
            raise ValueError(f'Comparisons use different mixtures/references: {baseline_key}')
        level = dbfs(audio['reference'])
        if level < -45:
            excluded.append(dict(identity, reason='reference_dbfs < -45', reference_dbfs=level, **hashes))
            continue
        with torch.inference_mode():
            prediction = float(si_sdr_value(audio['estimate'][None], audio['reference'][None]).mean())
            baseline = float(si_sdr_value(audio['mixture'][None], audio['reference'][None]).mean())
        results.append(dict(identity, si_sdr=prediction, mix_si_sdr=baseline, si_sdri=prediction-baseline,
                            reference_dbfs=level, frames=audio['reference'].shape[-1], **hashes))
    summary = summarize(results)
    report = {'metric': 'canonical centered channel-wise SI-SDR; mean channel dB, float32 CPU',
              'aggregation': 'one equally weighted retained track/role pair', 'sample_rate': 44100,
              'silence_threshold_dbfs': -45, 'torch_threads': 2, 'torch_version': torch.__version__,
              'manifest_sha256': sha256(manifest), 'retained_pairs': len(results),
              'excluded_pairs': excluded, 'summaries': summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.evaluate-', dir=output.parent) as scratch:
        staging = Path(scratch) / 'result'
        staging.mkdir()
        fields = list(IDENTITY) + ['si_sdr', 'mix_si_sdr', 'si_sdri', 'reference_dbfs', 'frames',
                                   'mixture_sha256', 'reference_sha256', 'estimate_sha256']
        with (staging / 'metrics_per_item.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        (staging / 'metrics_overall.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.output)
    print(f"Scored {report['retained_pairs']} pairs; excluded {len(report['excluded_pairs'])}; saved {args.output}")


if __name__ == '__main__':
    main()
