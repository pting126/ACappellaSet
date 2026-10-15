"""Portable K/L/M training with audited math, deterministic sampling, and resume."""
import importlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.present import DeterministicEpochSampler, keyed_seed
from models.conditioned import build_model
from models.query import ROLE_VOCAB
from models.registry import ROOT
from prepare_data import sha256
from .conditioned_math import (apply_v2a_augmentation, load_pretrained_htdemucs,
                               present_query_loss, verify_pt6s_load_contract)


def atomic_save(payload, path):
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    return {'torch':torch.get_rng_state(), 'python':random.getstate(),
            'numpy':np.random.get_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('A CUDA training checkpoint requires CUDA to resume')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def optimizer_for(model, recipe):
    embedding = [] if model.embedding_encoder is None else [
        p for p in model.embedding_encoder.parameters() if p.requires_grad]
    identities = {id(p) for p in embedding}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in identities]
    groups = [{'params':base,'lr':recipe['lr'],'name':'separator'}]
    if embedding:
        groups.append({'params':embedding,'lr':recipe['speaker_encoder_lr'],'name':'speaker_encoder'})
    return torch.optim.AdamW(groups, weight_decay=recipe['weight_decay'])


def initialize(config, recipe, initializer, encoder_initializer, output, device):
    from models.vendor.demucs import htdemucs
    original = htdemucs.HTDemucs

    def paired_constructor(*args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(recipe['seed'])
            return original(*args, **kwargs)

    htdemucs.HTDemucs = paired_constructor
    try:
        model = build_model(config, device)
    finally:
        htdemucs.HTDemucs = original
    if model.role_embedding is not None:
        generator = torch.Generator(device='cpu').manual_seed(recipe['seed'])
        weight = torch.randn(model.role_embedding.weight.shape, generator=generator)*.02
        for role in ('None','Bar','Lead'):
            weight[ROLE_VOCAB.index(role)].zero_()
        with torch.no_grad():
            model.role_embedding.weight.copy_(weight.to(model.role_embedding.weight))
    verify_pt6s_load_contract(load_pretrained_htdemucs(model, initializer, output, device))
    if model.embedding_encoder is not None:
        if encoder_initializer is None or sha256(encoder_initializer) != recipe['encoder_initialization_sha256']:
            raise ValueError('The registered original ReDimNet2 initializer is required')
        payload = torch.load(encoder_initializer, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload)
        result = model.embedding_encoder.model.load_state_dict(state, strict=False)
        # The reference loader permits projection extras; reject any other surprises.
        missing = [key for key in result.missing_keys if not key.startswith('frontend.') and 'projection' not in key]
        unexpected = [key for key in result.unexpected_keys if 'projection' not in key]
        if missing or unexpected:
            raise ValueError(f'Encoder initialization coverage mismatch: {missing}, {unexpected}')
        (output/'encoder_initialization.json').write_text(json.dumps({
            'sha256':recipe['encoder_initialization_sha256'], 'missing':result.missing_keys,
            'unexpected':result.unexpected_keys},indent=2)+'\n')
    return model


def _seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


def run(options, spec):
    if options.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    recipe_path = ROOT/'configs/training'/f"{spec['research_id']}.json"
    recipe = json.loads(recipe_path.read_text())
    config = json.loads((ROOT/spec['config']).read_text())
    if sha256(ROOT/spec['config']) != spec['config_sha256']:
        raise ValueError('Model config checksum mismatch')
    if sha256(ROOT/config['encoder_config']) != config['encoder_config_sha256']:
        raise ValueError('Encoder config checksum mismatch')
    initial = json.loads((ROOT/'third_party/initialization.json').read_text())
    if sha256(options.initialization) not in (initial['official_sha256'],initial['research_initializer_sha256']):
        raise ValueError('PT6S initializer checksum mismatch')
    for path in (options.manifest, options.valid_jc, options.valid_sk):
        if path is None or not path.is_dir():
            raise ValueError('--manifest, --valid-jc, and --valid-sk must be prepared directories')
        receipt = json.loads((path/'preparation.json').read_text())
        if receipt['metadata_only']:
            raise ValueError('Run preparation with audio verification before training')
        for name, digest in receipt['outputs'].items():
            if sha256(path/name) != digest:
                raise ValueError('Prepared manifest changed: '+name)
    if not options.cpu and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --cpu only for diagnostic checks')
    device = torch.device('cpu' if options.cpu else f'cuda:{options.device}')
    if device.type=='cuda': torch.cuda.set_device(device)
    if options.max_updates is not None and not 1 <= options.max_updates <= recipe['steps']:
        raise ValueError('--max-updates must be within the registered training budget')
    stop = options.max_updates or recipe['steps']
    output = options.output.absolute()
    if options.resume is None and output.exists():
        raise ValueError('Use a new output directory for fresh training')
    if options.resume is not None and not output.is_dir():
        raise ValueError('Resume requires the existing run output directory')
    identity = {'recipe_sha256':sha256(recipe_path),'config_sha256':spec['config_sha256'],
                'model':spec['research_id'], 'manifest_sha256':sha256(options.manifest/'DATA_PROTOCOL.json'),
                'valid_jc_sha256':sha256(options.valid_jc/'preparation.json'),
                'valid_sk_sha256':sha256(options.valid_sk/'preparation.json'),
                'deterministic':options.deterministic, 'torch_version':torch.__version__,
                'code_sha256':{str(path.relative_to(ROOT)):sha256(path)
                    for folder in ('training','data','models','evaluation')
                    for path in sorted((ROOT/folder).rglob('*.py'))}}
    saved = None
    if options.resume is not None:
        record = json.loads((output/'checkpoint.json').read_text())
        if options.resume.resolve() != (output/record['file']).resolve() or sha256(options.resume)!=record['sha256']:
            raise ValueError('Resume must use this run\'s verified latest checkpoint')
        saved = torch.load(options.resume,map_location='cpu',weights_only=False)
        if saved['identity']!=identity or saved['scheduler_state'] is not None:
            raise ValueError('Resume recipe, code, manifests, or runtime contract changed')
    if options.dry_run:
        print(json.dumps(dict(identity, updates=stop, device=str(device)),indent=2))
        return
    output.mkdir(parents=True,exist_ok=options.resume is not None)
    torch.manual_seed(recipe['seed']); random.seed(recipe['seed']); np.random.seed(recipe['seed'])
    model = initialize(config, recipe, options.initialization, options.speaker_initialization, output, device)
    dataset_class = importlib.import_module('data.strict_'+recipe['dataset_policy']).StrictPresentDataset
    dataset = dataset_class(options.manifest, 'train', config['conditioning'], channels='stereo',
                            seed=recipe['seed'], target_seconds=10., enroll_seconds=10.)
    sampler = DeterministicEpochSampler(dataset, recipe['seed'])
    generator = torch.Generator().manual_seed(keyed_seed(recipe['seed'],0,0,'train_worker_seed'))
    loader = DataLoader(dataset,batch_size=recipe['batch_size'],sampler=sampler,
                        num_workers=options.workers,pin_memory=device.type=='cuda',drop_last=True,
                        worker_init_fn=_seed_worker,generator=generator)
    if not len(loader):
        raise ValueError('Training population is smaller than one batch')
    optimizer = optimizer_for(model, recipe)
    step = micro_step = epoch = next_batch = 0
    if saved is not None:
        model.load_state_dict(saved['model_state'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        step,micro_step,epoch,next_batch = [saved[k] for k in ('step','micro_step','epoch','next_batch')]
        generator.set_state(saved['loader_generator_state'])
        restore_rng(saved['rng_state'])
        del saved
    if stop <= step:
        raise ValueError('Requested stopping update does not advance this run')
    (output/'launch.json').write_text(json.dumps(dict(identity,updates=stop,device=str(device)),indent=2)+'\n')
    augmentation = SimpleNamespace(**recipe)
    optimizer.zero_grad(set_to_none=True)
    from .validation import validate_and_select

    def save():
        if micro_step % recipe['grad_accum_steps']:
            raise RuntimeError('Checkpoints must be saved at optimizer-step boundaries')
        saved_epoch = epoch + int(next_batch >= len(loader))
        saved_batch = 0 if next_batch >= len(loader) else next_batch
        loader_state = generator.get_state() if saved_batch == 0 else iteration_generator_state
        payload = {'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                   'scheduler_state':None,'identity':identity,'step':step,'micro_step':micro_step,
                   'epoch':saved_epoch,'next_batch':saved_batch,'rng_state':rng_state(),
                   'loader_generator_state':loader_state}
        atomic_save(payload,output/'latest.pt')
        (output/'checkpoint.json').write_text(json.dumps({'file':'latest.pt',
            'sha256':sha256(output/'latest.pt'),'step':step},indent=2)+'\n')

    while step < stop:
        dataset.set_epoch(epoch); sampler.set_epoch(epoch)
        model.train()
        iteration_generator_state = generator.get_state()
        for batch_index,batch in enumerate(loader):
            if batch_index < next_batch: continue
            mixture,target,enrollment = [batch[key].to(device) for key in ('mixture','target','enrollment')]
            indices = [int(v) for v in batch['metadata']['example_index']]
            mixture,target,enrollment = apply_v2a_augmentation(mixture,target,enrollment,indices,epoch,augmentation)
            roles = torch.tensor([ROLE_VOCAB.index(v) for v in batch['metadata']['part']],device=device)
            prediction = model(mixture,enrollment,roles)
            loss,terms = present_query_loss(prediction,mixture,target,recipe['residual_loss_weight'],
                                           recipe['reconstruction_loss_weight'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            (loss/recipe['grad_accum_steps']).backward()
            micro_step += 1
            next_batch = batch_index+1
            if micro_step % recipe['grad_accum_steps']: continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['clip_grad_norm'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            with (output/'history.jsonl').open('a') as stream:
                stream.write(json.dumps({'step':step,'epoch':epoch,'loss':float(loss.detach()),
                                         'grad_norm':float(norm)})+'\n')
            if step % recipe['fullsong_valid_every']==0:
                state=rng_state()
                validate_and_select(model,config,spec,options.valid_jc,options.valid_sk,output,step,device)
                restore_rng(state);model.train()
            if step % recipe['paper_save_every']==0:
                atomic_save({'model_state':model.state_dict(),'config':config,'step':step,
                             'identity':identity},output/f'step{step:07d}.pt')
            if step % recipe['fullsong_valid_every']==0 or step==stop: save()
            if step==stop:break
        if next_batch >= len(loader): epoch+=1;next_batch=0
    atomic_save({'model_state':model.state_dict(),'step':step},output/'inference.pt')
    (output/'inference.json').write_text(json.dumps({
        'format':'acappellaset-conditioned-training-v1','model':spec['id'],
        'config_sha256':spec['config_sha256'],'step':step,'file':'inference.pt',
        'sha256':sha256(output/'inference.pt'),
        'note':'New training run; not a published paper checkpoint'},indent=2)+'\n')
    print(json.dumps({'status':'STOPPED_AT_REQUESTED_UPDATE','step':step,'output':str(output)}))
