"""Launch the original MSS fixed/specialist recipe from a verified patched checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from models.registry import ROOT, resolve


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='Fixed or original specialist model ID; selects its recipe, not its trained weights')
    parser.add_argument('--mss-root',type=Path,required=True)
    parser.add_argument('--initialization',type=Path,required=True,help='Verified original pt_6s.th or official htdemucs_6s initializer')
    parser.add_argument('--train-data',type=Path,required=True,help='Prepared mixture/stem training directory')
    parser.add_argument('--valid-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New run directory; existing directories are rejected')
    parser.add_argument('--device',type=int,default=0,help='CUDA device index')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dry-run',action='store_true',help='Verify inputs and print command without starting training')
    args=parser.parse_args(argv)
    spec=resolve(args.model)
    if args.device<0 or args.workers<0:parser.error('Device and workers must be nonnegative')
    if spec['family'] not in ('fixed-htdemucs','part-specific-htdemucs'):
        parser.error('Conditioned training is not supported by this MSS entry point')
    root=args.mss_root.resolve();output=args.output.absolute()
    if output.exists() or output.is_symlink():parser.error('Output must be a new directory')
    patch=json.loads((ROOT/'third_party/patches/mss-training.json').read_text())
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=patch['base_commit']:parser.error('MSS base commit differs from the registered recipe')
    for item in patch['files']:
        if sha256(root/item['path'])!=item['patched_sha256']:
            parser.error(f"MSS patch not verified: {item['path']}; run scripts/prepare_mss.py first")
    # Reject unrelated tracked changes rather than silently training a different implementation.
    changed=set(subprocess.check_output(['git','-C',str(root),'diff','HEAD','--name-only'],text=True).splitlines())
    if changed!={item['path'] for item in patch['files']}:
        parser.error('MSS has changes outside the reviewed patch')
    config=ROOT/spec['config']
    if sha256(config)!=spec['config_sha256']:parser.error('Model config checksum mismatch')
    initializer=json.loads((ROOT/'third_party/initialization.json').read_text())
    initial_hash=sha256(args.initialization)
    if initial_hash not in {initializer['official_sha256'],initializer['research_initializer_sha256']}:
        parser.error('Initializer checksum mismatch; this command does not accept arbitrary pickle checkpoints')
    for dataset in (args.train_data,args.valid_data):
        if not dataset.is_dir():parser.error(f'Dataset directory missing: {dataset}')
    if args.train_data.resolve()==args.valid_data.resolve():parser.error('Training and validation directories must differ')
    command=[sys.executable,str(root/'train.py'),'--model_type','htdemucs','--config_path',str(config),
             '--results_path',str(output),'--data_path',str(args.train_data.resolve()),
             '--valid_path',str(args.valid_data.resolve()),'--metrics','sdr','si_sdr',
             '--metric_for_scheduler','si_sdr','--loss','masked_loss','--num_workers',str(args.workers),
             '--device_ids',str(args.device),'--seed','0','--start_check_point',str(args.initialization.resolve()),
             '--wandb_offline']
    if spec['family']=='part-specific-htdemucs':command+=['--htdemucs_2stem_head_init','pt6s']
    print(shlex.join(command),flush=True)
    if args.dry_run:return
    output.mkdir(parents=True,exist_ok=False)
    receipt={'model_recipe':spec['id'],'upstream_commit':head,'patch_sha256':patch['patch_sha256'],
             'config_sha256':spec['config_sha256'],'initialization_sha256':initial_hash,'command':command,
             'note':'New training run, not a continuation or claim of exact paper reproduction'}
    (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    environment=dict(os.environ,WANDB_MODE='offline')
    result=subprocess.run(command,cwd=root,env=environment)
    receipt['returncode']=result.returncode
    (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
