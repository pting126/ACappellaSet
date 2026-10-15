"""Unchanged scientific functions extracted from the audited V2-A trainer."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
from data.present import keyed_seed

EPS = 1e-8
PT6S_EXPECTED_SKIPS = {
    'decoder.3.conv_tr.weight', 'decoder.3.conv_tr.bias',
    'tdecoder.3.conv_tr.weight', 'tdecoder.3.conv_tr.bias',
}

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


def target_and_residual(
    pred: Tensor,
    mixture: Tensor,
    target: Tensor,
    source_mode: str = "target_residual",
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    target_index = 1 if source_mode == "residual_target" else 0
    residual_index = 0 if source_mode == "residual_target" else 1
    est_target = pred[:, target_index]
    if pred.shape[1] < 2:
        return est_target, None, None
    est_residual = pred[:, residual_index]
    residual_target = mixture - target
    n = min(est_residual.shape[-1], residual_target.shape[-1])
    return est_target, est_residual[..., :n], residual_target[..., :n]


def _gain(rng: random.Random, low: float, high: float) -> float:
    return rng.uniform(low, high)


def apply_v2a_augmentation(
    mixture: Tensor,
    target: Tensor,
    enrollment: Tensor,
    indices: list[int],
    epoch: int,
    args,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use keyed per-example RNG streams shared by every V2-A cell."""
    if not args.augment_htdemucs_style:
        return mixture, target, enrollment
    mixture = mixture.clone()
    target = target.clone()
    enrollment = enrollment.clone()
    for batch_index, example_index in enumerate(indices):
        signal_rng = random.Random(
            keyed_seed(args.seed, epoch, example_index, "mixture_target_augmentation")
        )
        enrollment_rng = random.Random(
            keyed_seed(args.seed, epoch, example_index, "enrollment_augmentation")
        )
        residual = mixture[batch_index] - target[batch_index]
        if args.aug_loudness:
            target[batch_index] *= _gain(
                signal_rng, args.aug_loudness_min, args.aug_loudness_max
            )
            residual *= _gain(
                signal_rng, args.aug_loudness_min, args.aug_loudness_max
            )
            enrollment[batch_index] *= _gain(
                enrollment_rng, args.aug_loudness_min, args.aug_loudness_max
            )
            mixture[batch_index] = target[batch_index] + residual
        if mixture.shape[1] == 2 and signal_rng.random() < args.aug_channel_shuffle:
            mixture[batch_index] = mixture[batch_index].flip(0)
            target[batch_index] = target[batch_index].flip(0)
        if enrollment.shape[1] == 2 and enrollment_rng.random() < args.aug_channel_shuffle:
            enrollment[batch_index] = enrollment[batch_index].flip(0)
        if signal_rng.random() < args.aug_random_inverse:
            mixture[batch_index] = mixture[batch_index].flip(-1)
            target[batch_index] = target[batch_index].flip(-1)
        if enrollment_rng.random() < args.aug_random_inverse:
            enrollment[batch_index] = enrollment[batch_index].flip(-1)
        if signal_rng.random() < args.aug_random_polarity:
            mixture[batch_index] *= -1
            target[batch_index] *= -1
        if enrollment_rng.random() < args.aug_random_polarity:
            enrollment[batch_index] *= -1

    if mixture.shape[0] > 1 and args.aug_mixup:
        group_size = args.aug_mixup_group_size or mixture.shape[0]
        for start in range(0, mixture.shape[0], group_size):
            stop = min(start + group_size, mixture.shape[0])
            if stop - start < 2:
                continue
            group_indices = indices[start:stop]
            mixup_rng = random.Random(
                keyed_seed(args.seed, epoch, min(group_indices), "batch_mixup")
            )
            for probability in args.aug_mixup_probs:
                if mixup_rng.random() >= probability:
                    continue
                permutation = list(range(stop - start))
                mixup_rng.shuffle(permutation)
                residual = mixture[start:stop] - target[start:stop]
                gain = _gain(
                    mixup_rng,
                    args.aug_mixup_loudness_min,
                    args.aug_mixup_loudness_max,
                )
                mixture[start:stop] = (
                    target[start:stop] + residual + gain * residual[permutation]
                )
    return mixture, target, enrollment


def present_query_loss(
    prediction: Tensor,
    mixture: Tensor,
    target: Tensor,
    residual_weight: float = 0.25,
    reconstruction_weight: float = 0.10,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the exact V2-A mean of per-example present-query losses."""
    est_target, est_residual, residual_target = target_and_residual(
        prediction, mixture, target
    )
    if est_residual is None or residual_target is None:
        raise RuntimeError("V2-A requires explicit target and residual outputs")
    n = min(est_target.shape[-1], target.shape[-1])
    target_terms = neg_si_sdr(est_target[..., :n], target[..., :n]).mean(dim=1)
    residual_terms = neg_si_sdr(est_residual, residual_target).mean(dim=1)
    reconstruction_terms = (
        (prediction.sum(dim=1) - mixture).abs().mean(dim=(1, 2))
        / mixture.abs().mean(dim=(1, 2)).clamp_min(EPS)
    )
    per_example = (
        target_terms
        + residual_weight * residual_terms
        + reconstruction_weight * reconstruction_terms
    )
    return per_example.mean(), {
        "target_neg_si_sdr": target_terms.mean(),
        "residual_neg_si_sdr": residual_terms.mean(),
        "reconstruction_l1_normalized": reconstruction_terms.mean(),
    }


def load_pretrained_htdemucs(model: nn.Module, path: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        src_state = next((ckpt.get(key) for key in ("state_dict", "model_state_dict", "state", "best_state") if isinstance(ckpt.get(key), dict)), None)
        # Legacy Demucs releases saved the state dict directly as an OrderedDict.
        if src_state is None and ckpt and all(hasattr(value, "shape") for value in ckpt.values()):
            src_state = ckpt
    else:
        src_state = ckpt
    if not isinstance(src_state, dict):
        raise TypeError(f"unsupported htdemucs checkpoint object {type(src_state)}")
    own = model.demucs.state_dict()
    loadable = {}
    skipped = {}
    for key, value in src_state.items():
        clean = key.replace("module.", "")
        if clean in own and tuple(own[clean].shape) == tuple(value.shape):
            loadable[clean] = value
        elif clean in own:
            skipped[clean] = {"source": list(value.shape), "target": list(own[clean].shape)}
    missing, unexpected = model.demucs.load_state_dict(loadable, strict=False)
    report = {
        "checkpoint": str(path),
        "total_checkpoint_keys": len(src_state),
        "loaded_keys": len(loadable),
        "shape_skipped_keys": len(skipped),
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
        "shape_skipped_examples": dict(list(skipped.items())[:40]),
        "missing_examples": list(missing)[:40],
        "unexpected_examples": list(unexpected)[:40],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pretrained_htdemucs_load_report.json").write_text(json.dumps(report, indent=2))
    return report


def verify_pt6s_load_contract(report: dict[str, Any]) -> None:
    skipped = set(report["shape_skipped_examples"])
    if (
        report["total_checkpoint_keys"] != 525
        or report["loaded_keys"] != 521
        or skipped != PT6S_EXPECTED_SKIPS
        or report["unexpected_after_partial_load"] != 0
    ):
        raise RuntimeError(f"V2-A PT6S load contract failed: {json.dumps(report)}")
