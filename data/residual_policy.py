"""Shared S/K training-reference policy; never changes the extraction loss."""
import math

VERSION = 'sk_zero_residual_v1'
ENERGY_EPS = 1e-20
CHANNELS = 2


def exclude_channel_count(count, channels=CHANNELS):
    count = int(count)
    if channels != CHANNELS or count not in range(channels + 1):
        raise ValueError('Expected a valid stereo degenerate-channel count')
    return count == channels


def classify_energies(energies):
    values = [float(v) for v in energies]
    if len(values) != CHANNELS or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid stereo centered energies')
    count = sum(v <= ENERGY_EPS for v in values)
    return dict(zero_channels=count, exclude=exclude_channel_count(count),
                review_required=count == 1, centered_energy=values)


def centered_energies(audio):
    import torch
    if audio.ndim != 2 or audio.shape[0] != CHANNELS or audio.shape[1] < 2:
        raise ValueError('Expected stereo channel-first waveform')
    if not torch.isfinite(audio).all():
        raise ValueError('Nonfinite reference')
    return audio.double().var(dim=-1, correction=0).tolist()


def residual_decision(mixture, target):
    if mixture.shape != target.shape:
        raise ValueError('Mixture/target shape mismatch')
    return classify_energies(centered_energies(mixture - target))


def require_usable_references(mixture, target):
    decision = residual_decision(mixture, target)
    if decision['exclude'] or decision['review_required']:
        raise ValueError('SI-SDR-degenerate residual; must be excluded/reviewed before sampling')
    if classify_energies(centered_energies(target))['zero_channels']:
        raise ValueError('SI-SDR-degenerate target; review required')
    return decision
