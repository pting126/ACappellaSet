"""Full-song validation and six pair-weighted selection streams."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.enrollment import select_enrollment
from evaluation.metrics import dbfs, si_sdr_value
from evaluation.reporting import SHARED, summarize
from inference import read_enrollment
from models.conditioned import separate
from models.query import ROLE_VOCAB


def evaluate_model(model, config, spec, manifest, corpus, device, split='valid', allow_fallback=False):
    manifest = Path(manifest)
    with (manifest/f'eval_pairs_{split}_correct.csv').open(newline='') as stream:
        pairs = list(csv.DictReader(stream))
    with (manifest/'enrollment_index.csv').open(newline='') as stream:
        gallery = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in pairs:
        if row['split'] != split:
            raise ValueError('Evaluation manifest split mismatch')
        if corpus=='AB' and row['song_id']=='PC01': continue
        groups[row['track_id'],row['part']].append(row)
    scores, exclusions = [], []
    for (track, role), group in sorted(groups.items()):
        first = group[0]
        for field in ('song_id','mixture_path','target_path','singer_id'):
            if len({r[field] for r in group}) != 1:
                raise ValueError('Ambiguous evaluation pair: '+field)
        mixture, rate = sf.read(first['mixture_path'],dtype='float32',always_2d=True)
        reference, reference_rate = sf.read(first['target_path'],dtype='float32',always_2d=True)
        if rate!=44100 or reference_rate!=44100 or mixture.shape!=reference.shape or mixture.shape[1]!=2:
            raise ValueError('Expected aligned 44.1 kHz stereo evaluation audio')
        if not len(mixture) or not np.isfinite(mixture).all() or not np.isfinite(reference).all():
            raise ValueError('Empty or nonfinite evaluation audio')
        reference_tensor = torch.from_numpy(reference.T.copy())
        if dbfs(reference_tensor)<-45:
            exclusions.append({'track_id':track,'role':role,'reason':'silent_reference'})
            continue
        enrollment = None
        query = None
        if config['conditioning']!='role':
            query = select_enrollment(group,gallery,'max_active_different_song',config['enroll_seconds'])
            if query['pair_relation']!='different_song' and not allow_fallback:
                exclusions.append({'track_id':track,'role':role,'reason':'no_independent_enrollment'})
                continue
            enrollment,_ = read_enrollment(query['enroll_path'],44100,float(query['enroll_start']),config['enroll_seconds'])
        estimates = separate(model,config,mixture.T,device,enrollment,ROLE_VOCAB.index(role))
        score = float(si_sdr_value(torch.from_numpy(estimates[0])[None],reference_tensor[None]).mean())
        baseline = float(si_sdr_value(torch.from_numpy(mixture.T.copy())[None],reference_tensor[None]).mean())
        protocol = 'part_query' if config['conditioning']=='role' else (
            'fallback_inclusive' if allow_fallback else 'strict_enrollment')
        scores.append({'model':spec['id'],'corpus':corpus,'protocol':protocol,'song_id':first['song_id'],
                       'track_id':track,'role':role,'si_sdr':score,'mix_si_sdr':baseline,'si_sdri':score-baseline,
                       'query':query})
    summarize(scores)
    return scores, exclusions


def scope_scores(rows):
    result = {}
    for domain in ('jc','sk','combined'):
        selected = [r for r in rows if domain=='combined' or r['corpus'].lower()==domain]
        for scope in ('shared','inclusive'):
            retained = [r for r in selected if scope=='inclusive' or r['role'] in SHARED]
            if not retained: raise ValueError('Empty validation selection stream')
            result[domain+'_'+scope] = {'n_pairs':len(retained),
                'si_sdri':sum(r['si_sdri'] for r in retained)/len(retained)}
    return result


def validate_and_select(model,config,spec,jc_manifest,sk_manifest,output,step,device):
    from .conditioned import atomic_save
    rows = []
    for corpus, manifest in [('JC',jc_manifest),('SK',sk_manifest)]:
        scores, exclusions = evaluate_model(model,config,spec,manifest,corpus,device)
        if len(scores)!={'JC':60,'SK':52}[corpus]:
            raise ValueError(f'Historical validation cohort changed: {corpus}, {len(scores)} pairs')
        rows.extend(scores)
    scopes = scope_scores(rows)
    for name,expected in [('jc_shared',50),('sk_shared',50),('combined_shared',100),('combined_inclusive',112)]:
        if scopes[name]['n_pairs']!=expected: raise ValueError('Validation role population changed')
    destination = output/'validation'/f'step{step:07d}'
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'per_pair.json').write_text(json.dumps(rows,indent=2)+'\n')
    (destination/'scopes.json').write_text(json.dumps(scopes,indent=2)+'\n')
    selection_path = output/'selection.json'
    selections = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    for name,score in scopes.items():
        if name not in selections or score['si_sdri']>selections[name]['si_sdri']:
            filename='best_'+name+'.pt'
            atomic_save({'model_state':model.state_dict(),'config':config,'step':step,
                         'research_id':spec['research_id'],'selection':name},output/filename)
            from prepare_data import sha256
            selections[name]=dict(score,step=step,file=filename,sha256=sha256(output/filename))
            (output/('best_'+name+'.json')).write_text(json.dumps({
                'format':'acappellaset-conditioned-training-v1','model':spec['id'],
                'config_sha256':spec['config_sha256'],'step':step,'file':filename,
                'sha256':selections[name]['sha256'],'validation_selection':name,
                'note':'New training run; not a published paper checkpoint'},indent=2)+'\n')
    selection_path.write_text(json.dumps(selections,indent=2)+'\n')
