"""Unchanged canonical research SI-SDR and reference-level functions."""

import torch
from torch import Tensor

EPS = 1e-8


def neg_si_sdr(est: Tensor, target: Tensor, eps: float = EPS) -> Tensor:
    n = min(est.shape[-1], target.shape[-1])
    est = est[..., :n]
    target = target[..., :n]
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    proj = torch.sum(est * target, dim=-1, keepdim=True) * target / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    noise = est - proj
    ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps)


def si_sdr_value(est: Tensor, target: Tensor) -> Tensor:
    return -neg_si_sdr(est, target)


def dbfs(audio: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(audio.float() ** 2)).clamp_min(EPS)
    return float(20.0 * torch.log10(rms))
