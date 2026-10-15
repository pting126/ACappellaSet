"""Research audio loading helpers for verified 44.1 kHz manifests."""
import csv
from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf
import torch

EPS = 1e-8

def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def segments_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) < min(a1, b1)


def load_audio_segment(
    path: str,
    start: float,
    end: float,
    channels: str = "mono",
) -> torch.Tensor:
    info = sf.info(path)
    frame_offset = int(round(start * info.samplerate))
    frames = int(round((end - start) * info.samplerate))
    audio, _ = sf.read(path, start=frame_offset, frames=frames, always_2d=True, dtype="float32")
    if audio.shape[0] < frames:
        pad = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)
    # soundfile returns [time, channels]; torch convention is [channels, time].
    audio = torch.from_numpy(audio.T.copy())
    if channels == "mono":
        audio = audio.mean(dim=0, keepdim=True)
    elif channels == "stereo":
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
    else:
        raise ValueError(f"Unsupported channels mode: {channels}")
    return audio
