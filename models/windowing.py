"""Exact research full-song window coverage."""

def window_starts(total_frames: int, window_frames: int, hop_frames: int) -> list[int]:
    """Return starts that cover every frame and include an end-aligned window."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if hop_frames <= 0 or hop_frames > window_frames:
        raise ValueError("hop_frames must be in [1, window_frames]")
    if total_frames <= window_frames:
        return [0]
    starts = list(range(0, total_frames - window_frames + 1, hop_frames))
    final_start = total_frames - window_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
