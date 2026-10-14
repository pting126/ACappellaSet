"""Strict loading and MSS-compatible inference for fixed-output HTDemucs.

Chunking is adapted from ZFTurbo/Music-Source-Separation-Training
utils/model_utils.py (MIT; see third_party/licenses/MSS-MIT.txt).
The vendored architecture is unchanged from the audited MSS source.
"""

from contextlib import nullcontext
import hashlib
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from .demucs4ht import get_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint, config_path, spec, device):
    """Require the registered checkpoint/config identity before deserialization."""
    if sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the registered model.")
    if sha256(config_path) != spec["config_sha256"]:
        raise ValueError("Configuration SHA-256 differs from the registered model.")
    config = OmegaConf.load(config_path)
    sources = list(config.training.instruments)
    if sources != spec["source_order"]:
        raise ValueError("Configuration source order differs from the registry.")
    if (config.training.samplerate, config.training.channels) != (44100, 2):
        raise ValueError("This adapter requires the registered 44.1 kHz stereo model.")
    model = get_model(config)
    # Legacy training containers may contain objects beyond tensors. Only the
    # exact, maintainer-registered SHA-256 is accepted above; never silently
    # deserialize an arbitrary checkpoint or download unverified weights.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("state", "state_dict", "model_state_dict"):
            if key in payload:
                state = payload[key]
                break
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "module.")
    model.load_state_dict(state, strict=True)
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    if epoch is not None and int(epoch) != spec["epoch"]:
        raise ValueError("Checkpoint epoch differs from the registry.")
    config.inference.batch_size = spec["inference"]["batch_size"]
    config.training.use_amp = device.type == "cuda" and spec["inference"]["cuda_amp"]
    if config.training.segment != spec["inference"]["segment_seconds"]:
        raise ValueError("Unexpected segment duration.")
    if config.inference.num_overlap != spec["inference"]["num_overlap"]:
        raise ValueError("Unexpected overlap factor.")
    model.to(device).eval()
    return model, config


def separate(model, config, mixture, device):
    """Uniform overlap-add, right zero padding, and CPU accumulation as in MSS.

    Input is float audio [channels, samples]. Return [sources, channels, samples].
    No additional gain normalization or mixture-consistency projection is used.
    """
    mixture = np.asarray(mixture, dtype=np.float32)
    if mixture.ndim != 2 or mixture.shape[0] != config.training.channels:
        raise ValueError("Mixture must have shape [configured channels, samples].")
    if mixture.shape[-1] == 0 or not np.isfinite(mixture).all():
        raise ValueError("Mixture must be nonempty and contain only finite samples.")
    mix = torch.tensor(mixture, dtype=torch.float32)
    chunk_size = int(config.training.samplerate * config.training.segment)
    overlap = int(config.inference.num_overlap)
    batch_size = int(config.inference.batch_size)
    if overlap <= 0 or batch_size <= 0 or chunk_size // overlap <= 0:
        raise ValueError("Invalid chunk, overlap, or batch size.")
    step = chunk_size // overlap
    shape = (len(config.training.instruments),) + tuple(mix.shape)
    result = torch.zeros(shape, dtype=torch.float32)
    counter = torch.zeros(shape, dtype=torch.float32)
    amp = device.type == "cuda" and bool(config.training.use_amp)
    context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    with context, torch.inference_mode():
        batch, locations = [], []
        for start in range(0, mix.shape[-1], step):
            part = mix[:, start:start + chunk_size].to(device)
            length = part.shape[-1]
            batch.append(F.pad(part, (0, chunk_size - length), mode="constant", value=0))
            locations.append((start, length))
            if len(batch) >= batch_size or start + step >= mix.shape[-1]:
                prediction = model(torch.stack(batch, dim=0))
                expected = (len(batch), len(config.training.instruments), mix.shape[0], chunk_size)
                if tuple(prediction.shape) != expected:
                    raise ValueError(f"Unexpected model output shape: {tuple(prediction.shape)}")
                if not torch.isfinite(prediction).all():
                    raise ValueError("Model produced non-finite samples.")
                for index, (offset, length) in enumerate(locations):
                    result[..., offset:offset + length] += prediction[index, ..., :length].cpu()
                    counter[..., offset:offset + length] += 1.0
                batch.clear()
                locations.clear()
        estimated = (result / counter).numpy()
    if not np.isfinite(estimated).all():
        raise ValueError("Overlap-add produced non-finite samples.")
    return estimated
