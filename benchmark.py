"""Evaluate one frozen model on an exact prepared test corpus."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import tempfile

ROLES={'soprano':'S','alto':'A','tenor':'T','bass':'B','vp':'VP','lead_vocal':'Lead','baritone':'Bar'}


def evaluate_fixed(model,config,spec,manifest,corpus,device):
    import numpy as np
    import soundfile as sf
    import torch
    from models.fixed import separate
    from evaluation.metrics import dbfs,si_sdr_value
    from evaluation.reporting import summarize
    with (manifest/'eval_pairs_test_correct.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    tracks=defaultdict(dict)
    for row in rows:
        if row['split']!='test':raise ValueError('Expected a test manifest')
        if corpus=='AB' and row['song_id']=='PC01':continue
        previous=tracks[row['track_id']].get(row['part'])
        if previous and any(previous[k]!=row[k] for k in ('song_id','mixture_path','target_path')):
            raise ValueError('Conflicting source identity')
        tracks[row['track_id']][row['part']]=row
    outputs={}
    for index,stem in enumerate(spec['source_order']):
        role=ROLES.get(stem)
        if stem=='other' and spec['id']=='Fixed_6stem_htdemucs_SKnJC':role='Lead' if corpus=='JC' else 'Bar'
        if role:outputs[role]=index
    results,exclusions=[],[]
    for track,parts in sorted(tracks.items()):
        if len({r['mixture_path'] for r in parts.values()})!=1:raise ValueError('Conflicting mixtures for one track')
        first=next(iter(parts.values()))
        mixture,rate=sf.read(first['mixture_path'],dtype='float32',always_2d=True)
        if rate!=44100 or mixture.ndim!=2 or mixture.shape[1]!=2 or not len(mixture) or not np.isfinite(mixture).all():
            raise ValueError('Expected finite 44.1 kHz stereo mixture')
        estimates=separate(model,config,mixture.T,device)
        for role,row in sorted(parts.items()):
            if role not in outputs:
                exclusions.append({'track_id':track,'role':role,'reason':'no_matching_output'});continue
            reference,ref_rate=sf.read(row['target_path'],dtype='float32',always_2d=True)
            if ref_rate!=rate or reference.shape!=mixture.shape or not np.isfinite(reference).all():
                raise ValueError('Reference is not aligned, finite stereo audio')
            ref=torch.from_numpy(reference.T.copy())
            if dbfs(ref)<-45:
                exclusions.append({'track_id':track,'role':role,'reason':'silent_reference'});continue
            estimate=estimates[outputs[role]]
            if estimate.shape!=reference.T.shape or not np.isfinite(estimate).all():raise ValueError('Invalid estimate')
            score=float(si_sdr_value(torch.from_numpy(estimate)[None],ref[None]).mean())
            baseline=float(si_sdr_value(torch.from_numpy(mixture.T.copy())[None],ref[None]).mean())
            results.append(dict(model=spec['id'],corpus=corpus,protocol='mixture_only',song_id=row['song_id'],
                                track_id=track,role=role,si_sdr=score,mix_si_sdr=baseline,si_sdri=score-baseline))
    summarize(results)
    return results,exclusions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--corpus',choices=['JC','SK','AB'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--allow-enrollment-fallback',action='store_true',help='Explicit AB fallback-inclusive evaluation')
    parser.add_argument('--allow-untrained-role',action='store_true',help='Explicit zero-role diagnostic for conditioned part/dual queries')
    args=parser.parse_args()
    from prepare_data import sha256
    from models.registry import ROOT,resolve,select_checkpoint
    if args.output.exists() or args.output.is_symlink():parser.error('Output must be a new directory')
    spec=select_checkpoint(resolve(args.model),args.selection)
    receipt=json.loads((args.manifest/'preparation.json').read_text())
    if receipt['metadata_only'] or receipt['set']!='test_'+args.corpus.lower():
        parser.error('Use the matching audio-verified test manifest')
    for name,digest in receipt['outputs'].items():
        if sha256(args.manifest/name)!=digest:parser.error('Prepared test manifest changed: '+name)
    conditioned=spec['family']=='conditioned-htdemucs'
    if not conditioned and (args.allow_enrollment_fallback or args.allow_untrained_role):
        parser.error('These diagnostic options require a conditioned model')
    if args.allow_enrollment_fallback and args.corpus!='AB':parser.error('Fallback-inclusive is an AB diagnostic')
    import torch
    from evaluation.reporting import summarize
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(1337)
    device=torch.device(args.device)
    if conditioned:
        from models.conditioned import load_model
        from training.validation import evaluate_model
        model,config=load_model(args.checkpoint,ROOT/spec['config'],spec,device)
        allowed=None if config['conditioning']=='enrollment' or args.allow_untrained_role else config['training_roles']
        rows,excluded=evaluate_model(model,config,spec,args.manifest,args.corpus,device,split='test',
                                    allow_fallback=args.allow_enrollment_fallback,allowed_roles=allowed)
    else:
        from models.fixed import load_model
        model,config=load_model(args.checkpoint,ROOT/spec['config'],spec,device)
        rows,excluded=evaluate_fixed(model,config,spec,args.manifest,args.corpus,device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.benchmark-',dir=args.output.parent) as temp:
        staging=Path(temp)/'result';staging.mkdir()
        (staging/'per_pair.json').write_text(json.dumps(rows,indent=2)+'\n')
        (staging/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')
        (staging/'excluded.json').write_text(json.dumps(excluded,indent=2)+'\n')
        (staging/'provenance.json').write_text(json.dumps(dict(model=spec['id'],selection=args.selection,
            checkpoint_sha256=sha256(args.checkpoint),config_sha256=spec['config_sha256'],
            manifest_sha256=sha256(args.manifest/'preparation.json'),device=str(device),torch=torch.__version__,
            allow_enrollment_fallback=args.allow_enrollment_fallback,allow_untrained_role=args.allow_untrained_role,
            note='Checkpoint selection is supplied before testing; this command does not reselect weights'),indent=2)+'\n')
        staging.rename(args.output)
    print(f'Scored {len(rows)} retained pairs; {len(excluded)} excluded.')


if __name__=='__main__':main()
