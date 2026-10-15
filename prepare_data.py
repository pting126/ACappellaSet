"""Materialize exact historical manifests against user-provided audio roots."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_roots(values):
    roots = {}
    for value in values:
        key, separator, path = value.partition('=')
        if not separator or key not in ('SK','JC','AB') or not path or key in roots:
            raise ValueError('Use each --root once, as SK=/path, JC=/path, or AB=/path')
        roots[key] = Path(path).resolve()
    return roots


def materialize(name, roots, output, metadata_only=False):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output must be a new directory')
    registry = json.loads((ROOT/'manifests/splits/index.json').read_text())
    if name not in registry['sets']:
        raise ValueError('Unknown manifest set')
    entry = registry['sets'][name]
    rows_by_file = {}
    payloads = {}
    for filename, info in entry['files'].items():
        source = ROOT/info['file']
        if sha256(source) != info['sha256']:
            raise ValueError('Published manifest checksum mismatch: '+filename)
        if filename.endswith('.csv'):
            raw = gzip.decompress(source.read_bytes())
            if hashlib.sha256(raw).hexdigest() != info['content_sha256']:
                raise ValueError('Manifest content checksum mismatch')
            reader = csv.DictReader(io.StringIO(raw.decode()))
            rows = list(reader)
            if len(rows) != info['rows']:
                raise ValueError('Manifest population count changed')
            for row in rows:
                for key, value in row.items():
                    tag, separator, relative = value.partition('/')
                    if separator and tag in ('SK','JC','AB'):
                        if tag not in roots:
                            raise ValueError('Missing --root '+tag+'=/path')
                        if '..' in Path(relative).parts or Path(relative).is_absolute():
                            raise ValueError('Invalid relative audio path')
                        row[key] = str(roots[tag]/relative)
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
            payloads[filename] = stream.getvalue().encode()
            rows_by_file[filename] = rows
        else:
            payloads[filename] = source.read_bytes()
    verified_audio = {}
    if not metadata_only:
        import soundfile as sf
        needed = {}
        for filename, rows in rows_by_file.items():
            if filename == 'excluded_pairs.csv':
                continue
            for row in rows:
                for field in ('mixture_path', 'target_path', 'enroll_path'):
                    path = row.get(field)
                    if not path:
                        continue
                    expected = row.get('source_audio_sha256') if field == 'target_path' else None
                    if field == 'enroll_path' and filename == 'enrollment_index.csv':
                        expected = row.get('source_audio_sha256')
                    if expected and needed.get(path) and needed[path] != expected:
                        raise ValueError('Conflicting audio hashes: '+path)
                    needed[path] = expected or needed.get(path)
        for path, expected in sorted(needed.items()):
            info = sf.info(path)
            if info.samplerate != 44100 or info.channels != 2 or info.frames < 1:
                raise ValueError('Expected nonempty 44.1 kHz stereo audio: '+path)
            digest = sha256(path)
            if expected and digest != expected:
                raise ValueError('Historical audio hash mismatch: '+path)
            verified_audio[path] = {'sha256': digest, 'frames': info.frames,
                                    'historical_hash_checked': bool(expected)}
        for rows in rows_by_file.values():
            for row in rows:
                mix, target = row.get('mixture_path'), row.get('target_path')
                if mix in verified_audio and target in verified_audio:
                    if verified_audio[mix]['frames'] != verified_audio[target]['frames']:
                        raise ValueError('Mixture/reference length mismatch')
    if 'DATA_PROTOCOL.json' in payloads:
        policy = json.loads(payloads['DATA_PROTOCOL.json'])
        policy['target_sha256'] = hashlib.sha256(payloads['target_examples.csv']).hexdigest()
        policy['enrollment_sha256'] = hashlib.sha256(payloads['enrollment_index.csv']).hexdigest()
        payloads['DATA_PROTOCOL.json'] = (json.dumps(policy,indent=2)+'\n').encode()
        payloads['COMPLETE.json'] = (json.dumps({'protocol_sha256':hashlib.sha256(
            payloads['DATA_PROTOCOL.json']).hexdigest()})+'\n').encode()
    receipt = {'set':name,'metadata_only':metadata_only,'source_files':entry['files'],
               'roots':{k:str(v) for k,v in roots.items()},'audio':verified_audio,
               'outputs':{k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},
               'note':'Only filesystem references changed; row order and scientific fields retained'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.acappellaset-data-',dir=output.parent) as temp:
        staging = Path(temp)/'manifest'
        staging.mkdir()
        for filename, raw in payloads.items():
            (staging/filename).write_bytes(raw)
        (staging/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if output.exists():
            raise FileExistsError('Output appeared during preparation')
        staging.rename(output)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',required=True,help='Registered historical set, e.g. train_sk, valid_jc, test_sk')
    parser.add_argument('--root',action='append',default=[],help='Corpus root, e.g. SK=/path/to/6_stem')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata-only',action='store_true',help='Materialize paths without claiming audio readiness')
    args=parser.parse_args()
    result=materialize(args.set,parse_roots(args.root),args.output,args.metadata_only)
    print(json.dumps({'set':result['set'],'verified_audio_files':len(result['audio']),
                      'metadata_only':result['metadata_only']}))


if __name__=='__main__':main()
