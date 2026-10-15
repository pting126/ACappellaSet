#!/usr/bin/env python3
"""Deterministic present-query dataset used by the V2-A ablation."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .audio import (
    EPS,
    load_audio_segment,
    read_csv,
    segments_overlap,
)


def keyed_seed(global_seed: int, epoch: int, index: int, purpose: str) -> int:
    text = f"v2a:{global_seed}:{epoch}:{index}:{purpose}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class DeterministicEpochSampler(Sampler[int]):
    """Shuffle deterministically without depending on worker or RNG call order."""

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.Random(keyed_seed(self.seed, self.epoch, 0, "sample_order")).shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class V2APresentDataset(Dataset):
    """Return one active target and, when needed, a same-singer enrollment."""

    def __init__(
        self,
        manifest_dir: Path | str,
        split: str,
        conditioning: str,
        channels: str = "stereo",
        seed: int = 1337,
        target_seconds: float = 10.0,
        enroll_seconds: float = 10.0,
        max_examples: int | None = None,
        target_parts: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if conditioning not in {"none", "role", "enrollment", "dual"}:
            raise ValueError(f"unsupported V2-A conditioning: {conditioning}")
        self.manifest_dir = Path(manifest_dir)
        self.split = split
        self.conditioning = conditioning
        self.channels = channels
        self.seed = seed
        self.target_seconds = target_seconds
        self.enroll_seconds = enroll_seconds
        self.epoch = 0
        accepted_parts = set(target_parts or ())
        all_rows = read_csv(self.manifest_dir / "target_examples.csv")
        self.rows = []
        for manifest_index, row in enumerate(all_rows):
            if row["split"] != split:
                continue
            if accepted_parts and row["part"] not in accepted_parts:
                continue
            tagged = dict(row)
            tagged["_v2a_example_index"] = manifest_index
            self.rows.append(tagged)
        if max_examples is not None:
            self.rows = self.rows[:max_examples]
        if not self.rows:
            raise RuntimeError(f"no V2-A examples for split={split}")
        self.enroll_by_singer: dict[str, list[dict]] = defaultdict(list)
        if conditioning not in {"none", "role"}:
            for row in read_csv(self.manifest_dir / "enrollment_index.csv"):
                if row["split"] == split:
                    self.enroll_by_singer[row["singer_id"]].append(row)
            missing = [
                row["example_id"] for row in self.rows
                if not self._enrollment_candidates(row)
            ]
            if missing:
                raise RuntimeError(
                    f"no correct enrollment for {len(missing)} examples; first={missing[:5]}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _enrollment_candidates(self, target: dict) -> list[dict]:
        candidates = []
        for enrollment in self.enroll_by_singer[target["singer_id"]]:
            same_file = enrollment["enroll_path"] == target["target_path"]
            overlaps = same_file and segments_overlap(
                float(enrollment["enroll_start"]),
                float(enrollment["enroll_end"]),
                float(target["target_start"]),
                float(target["target_end"]),
            )
            if not overlaps:
                candidates.append(enrollment)
        return candidates

    def _choose_enrollment(self, target: dict, index: int) -> dict:
        candidates = self._enrollment_candidates(target)
        different_song = [
            row for row in candidates if row["song_id"] != target["song_id"]
        ]
        pool = different_song or candidates
        rng = random.Random(keyed_seed(self.seed, self.epoch, index, "enrollment"))
        return pool[rng.randrange(len(pool))]

    def __getitem__(self, index: int) -> dict:
        target_row = self.rows[index]
        start = float(target_row["target_start"])
        end = start + self.target_seconds
        mixture = load_audio_segment(
            target_row["mixture_path"], start, end, channels=self.channels
        )
        target = load_audio_segment(
            target_row["target_path"], start, end, channels=self.channels
        )
        mix_scale = mixture.abs().max().clamp_min(EPS)
        mixture = mixture / mix_scale
        target = target / mix_scale

        enrollment_row = None
        if self.conditioning in {"none", "role"}:
            # Shape-compatible placeholder only. No enrollment file is decoded.
            enrollment = torch.zeros_like(target)
        else:
            enrollment_row = self._choose_enrollment(target_row, index)
            enroll_start = float(enrollment_row["enroll_start"])
            enrollment = load_audio_segment(
                enrollment_row["enroll_path"],
                enroll_start,
                enroll_start + self.enroll_seconds,
                channels=self.channels,
            )
            enrollment = enrollment / enrollment.abs().max().clamp_min(EPS)

        metadata = {
            "example_id": target_row["example_id"],
            "example_index": int(target_row["_v2a_example_index"]),
            "epoch": self.epoch,
            "track_id": target_row["track_id"],
            "song_id": target_row["song_id"],
            "part": target_row["part"],
            "singer_id": target_row["singer_id"],
            "target_path": target_row["target_path"],
            "target_start": target_row["target_start"],
            "target_end": target_row["target_end"],
            "query_status": "present",
        }
        if enrollment_row is not None:
            metadata.update({
                "enroll_id": enrollment_row["enroll_id"],
                "enroll_song_id": enrollment_row["song_id"],
                "enroll_singer_id": enrollment_row["singer_id"],
                "enroll_start": enrollment_row["enroll_start"],
            })
        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "metadata": metadata,
        }
