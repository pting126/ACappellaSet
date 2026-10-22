"""Read-only integrity checks for the public code and metadata package."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.registry import entries


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=0
    patch=json.loads((ROOT/'third_party/patches/mss-training.json').read_text())
    if digest(ROOT/'third_party/patches/mss-training.patch')!=patch['patch_sha256']:
        raise ValueError('MSS training patch checksum mismatch')
    checks+=1
    for spec in entries():
        if digest(ROOT/spec['config'])!=spec['config_sha256']:raise ValueError('Config changed: '+spec['id'])
        checks+=1
    index=json.loads((ROOT/'manifests/splits/index.json').read_text())
    for entry in index['sets'].values():
        for item in entry['files'].values():
            path=ROOT/item['file']
            if digest(path)!=item['sha256']:raise ValueError('Split manifest changed: '+str(path))
            if item.get('content_sha256') and hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()!=item['content_sha256']:
                raise ValueError('Decompressed manifest changed')
            checks+=1
    source=json.loads((ROOT/'third_party/conditioned_training_sources.json').read_text())
    for item in source['files']:
        if digest(ROOT/item['file'])!=item['sha256']:raise ValueError('Audited source changed: '+item['file'])
        checks+=1
    from reproduce import load_artifacts,build_tables
    plan,groups,summaries=load_artifacts();build_tables(plan,summaries)
    for path in ROOT.rglob('*.md'):
        if any(part in ('.git','.venv','outputs','runs') for part in path.relative_to(ROOT).parts):continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith(('mailto:','#')):continue
            filename=unquote(target.split('#')[0].split(' "')[0])
            if filename and not (path.parent/filename).exists():raise ValueError(f'Broken local link in {path.relative_to(ROOT)}: {target}')
    print(json.dumps({'status':'PASS','registered_recipes':len(entries()),'hash_checks':checks,
                      'split_sets':len(index['sets']),'score_artifacts':len(groups),'local_document_links':'PASS'}))


if __name__=='__main__':main()
