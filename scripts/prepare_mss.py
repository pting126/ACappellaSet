"""Apply the reviewed MSS patch to a clean checkout of its exact upstream base."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(checkout, check_only=False):
    checkout = Path(checkout).resolve()
    metadata = json.loads((ROOT/'third_party/patches/mss-training.json').read_text())
    patch = ROOT/'third_party/patches/mss-training.patch'
    if sha(patch.read_bytes()) != metadata['patch_sha256']:
        raise ValueError('Patch checksum mismatch')
    head = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if head != metadata['base_commit']:
        raise ValueError(f"Expected MSS commit {metadata['base_commit']}; found {head}")
    status = subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True)
    if status:
        raise ValueError('Use a clean, dedicated MSS checkout; local changes will not be overwritten')
    originals = {}
    for item in metadata['files']:
        path = checkout/item['path']
        if path.is_symlink():
            raise ValueError(f'Unexpected symlink: {path}')
        data = path.read_bytes()
        if sha(data) != item['upstream_sha256']:
            raise ValueError(f"Upstream file checksum mismatch: {item['path']}")
        originals[item['path']] = data
    # Check/apply in a temporary tree first. No partially applicable patch can
    # modify the user's checkout. Normalize only the six reviewed files.
    with tempfile.TemporaryDirectory(prefix='acappellaset-mss-') as tmp:
        staging = Path(tmp)
        for name,data in originals.items():
            path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data.replace(b'\r\n',b'\n'))
        subprocess.run(['git','apply','--check',str(patch)],cwd=staging,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=staging,check=True)
        patched = {}
        for item in metadata['files']:
            data=(staging/item['path']).read_bytes()
            if sha(data)!=item['patched_sha256']:
                raise ValueError(f"Patched checksum mismatch: {item['path']}")
            patched[item['path']]=data
        if not check_only:
            for name,data in originals.items():
                if (checkout/name).read_bytes()!=data:
                    raise ValueError('Checkout changed during patch preparation')
            try:
                for name,data in patched.items():
                    (checkout/name).write_bytes(data)
            except BaseException:
                for name,data in originals.items():
                    (checkout/name).write_bytes(data)
                raise
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    metadata=prepare(args.checkout,args.check_only)
    print(('Verified' if args.check_only else 'Applied')+f" patch for MSS {metadata['base_commit']}")


if __name__=='__main__':main()
