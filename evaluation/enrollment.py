"""Deterministic enrollment selection for full-song evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf


SELECTION_MODES = (
    "legacy_first",
    "max_active_different_song",
    "max_active_same_song",
)


@lru_cache(maxsize=None)
def activity_stats(
    path: str,
    start: float,
    duration: float,
    frame_seconds: float = 0.1,
    relative_threshold_db: float = 35.0,
    absolute_floor_dbfs: float = -60.0,
) -> tuple[float, float, float]:
    """Return active-frame fraction, threshold, and clip RMS in dBFS."""
    info = sf.info(path)
    start_frame = round(start * info.samplerate)
    frame_count = round(duration * info.samplerate)
    audio, _ = sf.read(
        path,
        start=start_frame,
        frames=frame_count,
        dtype="float32",
        always_2d=True,
    )
    mono = np.mean(audio, axis=1)
    expected = frame_count
    if mono.size < expected:
        mono = np.pad(mono, (0, expected - mono.size))
    frame_size = max(1, round(frame_seconds * info.samplerate))
    usable = (mono.size // frame_size) * frame_size
    frames = mono[:usable].reshape(-1, frame_size)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12))
    peak_frame_db = float(np.max(frame_db))
    threshold = max(peak_frame_db - relative_threshold_db, absolute_floor_dbfs)
    active_fraction = float(np.mean(frame_db >= threshold))
    clip_db = float(20.0 * np.log10(np.sqrt(np.mean(mono**2) + 1e-12)))
    return active_fraction, threshold, clip_db


def _legacy_selection(rows: list[dict[str, str]]) -> dict[str, Any]:
    target_song = rows[0]["song_id"]
    different_song = [row for row in rows if row["enroll_song_id"] != target_song]
    candidates = different_song or rows
    selected = min(
        candidates,
        key=lambda row: (
            row["enroll_song_id"],
            float(row["enroll_start"]),
            row["enroll_id"],
            row.get("example_id", ""),
        ),
    )
    return dict(selected)


def _normalize_index_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "enroll_id": row["enroll_id"],
        "enroll_path": row["enroll_path"],
        "enroll_start": row["enroll_start"],
        "enroll_end": row["enroll_end"],
        "enroll_track_id": row["track_id"],
        "enroll_song_id": row["song_id"],
        "enroll_part": row["part"],
        "enroll_singer_id": row["singer_id"],
        "enroll_energy_dbfs": row["enroll_energy_dbfs"],
        "enrollment_type": "correct",
    }


def select_enrollment(
    rows: list[dict[str, str]],
    enrollment_index: list[dict[str, str]],
    mode: str,
    enroll_seconds: float,
) -> dict[str, Any]:
    """Select one enrollment for a concrete target track/stem group."""
    if mode not in SELECTION_MODES:
        raise ValueError(f"unknown enrollment selection mode: {mode}")
    first = rows[0]
    if mode == "legacy_first":
        selected = _legacy_selection(rows)
        fallback = "none"
    else:
        same_singer = [
            row for row in enrollment_index if row["singer_id"] == first["singer_id"]
        ]
        want_same_song = mode == "max_active_same_song"
        preferred = [
            row
            for row in same_singer
            if (row["song_id"] == first["song_id"]) == want_same_song
        ]
        candidates = preferred or same_singer
        if not candidates:
            raise RuntimeError(
                f"no enrollment candidates for singer {first['singer_id']} in {first['track_id']}"
            )
        fallback = "none" if preferred else (
            "different_song" if want_same_song else "same_song"
        )

        def ranking(row: dict[str, str]) -> tuple[float, float, str, float, str]:
            active_fraction, _, clip_db = activity_stats(
                row["enroll_path"],
                float(row["enroll_start"]),
                enroll_seconds,
            )
            return (
                -active_fraction,
                -clip_db,
                row["song_id"],
                float(row["enroll_start"]),
                row["enroll_id"],
            )

        selected = _normalize_index_row(min(candidates, key=ranking))

    active_fraction, threshold, clip_db = activity_stats(
        selected["enroll_path"],
        float(selected["enroll_start"]),
        enroll_seconds,
    )
    same_song = selected["enroll_song_id"] == first["song_id"]
    same_track = selected["enroll_track_id"] == first["track_id"]
    if same_track:
        relation = "same_track_reference_excerpt"
    elif same_song:
        relation = "same_song_other_track"
    else:
        relation = "different_song"
    selected.update(
        {
            "pair_relation": relation,
            "enrollment_selection": mode,
            "enrollment_selection_fallback": fallback,
            "enroll_active_fraction": active_fraction,
            "enroll_activity_threshold_dbfs": threshold,
            "enroll_measured_dbfs": clip_db,
        }
    )
    return selected
