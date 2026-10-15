"""Strict conditioned checkpoint loading and native full-song overlap-add."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .fixed import sha256
from .registry import ROOT
from .windowing import window_starts


def build_model(config, device):
    from .conditioned_core import DualConditionedHTDemucs
    return DualConditionedHTDemucs(
        demucs_root=None, source_mode=config['source_mode'], audio_channels=2,
        emb_dim=config['emb_dim'], embedding_backend=config['embedding_backend'],
        freeze_external_embedding=config['freeze_external_embedding'],
        wespeaker_model_path=str(ROOT / config['encoder_config']),
        channels=config['ht_channels'], depth=config['ht_depth'], growth=config['ht_growth'],
        bottom_channels=config['ht_bottom_channels'], dconv_mode=config['ht_dconv_mode'],
        t_layers=config['ht_t_layers'], nfft=config['ht_nfft'], segment=config['seconds'],
        sample_rate=config['sample_rate'], role_embedding_dim=config['role_embedding_dim'],
        conditioning=config['conditioning']).to(device)


def load_model(checkpoint, config_path, spec, device):
    checkpoint = Path(checkpoint)
    if sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch; refusing unregistered pickle content')
    if sha256(config_path) != spec['config_sha256']:
        raise ValueError('Conditioned config checksum mismatch')
    config = json.loads(Path(config_path).read_text())
    if sha256(ROOT / config['encoder_config']) != config['encoder_config_sha256']:
        raise ValueError('Encoder configuration checksum mismatch')
    if config['source_mode'] != 'target_residual' or spec['source_order'] != ['target', 'residual']:
        raise ValueError('Unexpected conditioned output order')
    # Only exact, registered research checkpoints reach this legacy loader.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['step'] != spec['step']:
        raise ValueError('Checkpoint step does not match its registered selection')
    model = build_model(config, device)
    model.load_state_dict(payload['model_state'], strict=True)
    zero_role = spec.get('untrained_zero_role')
    if model.role_embedding is not None and zero_role is not None:
        token = model.role_embedding.weight[model.role_vocab.index(zero_role)]
        if torch.count_nonzero(token).item():
            raise ValueError('The historical untrained role token is not exactly zero')
        if model.role_spec_film.bias is not None or model.role_time_film.bias is not None:
            raise ValueError('Untrained-role contract requires bias-free role projections')
    model.eval()
    return model, config


def separate(model, config, mixture, device, enrollment=None, role_index=7):
    mixture = torch.as_tensor(np.ascontiguousarray(mixture), dtype=torch.float32)
    if mixture.ndim != 2 or mixture.shape[0] != 2 or mixture.shape[-1] < 1:
        raise ValueError('Expected nonempty stereo mixture')
    if not torch.isfinite(mixture).all():
        raise ValueError('Nonfinite mixture audio')
    rate = config['sample_rate']
    window = round(config['seconds'] * rate)
    hop = round(config['hop_seconds'] * rate)
    frames = mixture.shape[-1]
    if model.embedding_encoder is None:
        enrollment_tensor = torch.zeros(2, round(config['enroll_seconds'] * rate))
    else:
        if enrollment is None:
            raise ValueError('Enrollment audio is required')
        enrollment_tensor = torch.as_tensor(np.ascontiguousarray(enrollment), dtype=torch.float32)
        if enrollment_tensor.shape != (2, round(config['enroll_seconds'] * rate)):
            raise ValueError('Enrollment must be a prepared stereo segment of the registered duration')
        if not torch.isfinite(enrollment_tensor).all():
            raise ValueError('Nonfinite enrollment audio')
        enrollment_tensor = enrollment_tensor / enrollment_tensor.abs().max().clamp_min(1e-8)
    enrollment_tensor = enrollment_tensor.unsqueeze(0).to(device)
    query = torch.tensor([role_index], device=device)
    total = torch.zeros(2, 2, frames)
    weights_total = torch.zeros(1, frames)
    model.eval()
    with torch.no_grad():
        for start in window_starts(frames, window, hop):
            stop = min(start + window, frames)
            valid = stop - start
            chunk = mixture[:, start:stop]
            if valid < window:
                chunk = F.pad(chunk, (0, window - valid))
            scale = chunk.abs().max().clamp_min(1e-8)
            prediction = model((chunk / scale).unsqueeze(0).to(device), enrollment_tensor, query)
            estimate = prediction.squeeze(0).detach().cpu()[..., :valid] * scale
            if estimate.shape != (2, 2, valid) or not torch.isfinite(estimate).all():
                raise ValueError('Invalid conditioned output')
            weights = (torch.ones(1) if valid == 1 else
                       torch.hann_window(valid, periodic=False).clamp_min(1e-3))
            total[:, :, start:stop] += estimate * weights
            weights_total[:, start:stop] += weights
    if not bool((weights_total > 0).all()):
        raise RuntimeError('Overlap-add left uncovered samples')
    return (total / weights_total).numpy()
