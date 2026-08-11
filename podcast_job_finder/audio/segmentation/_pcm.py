from math import ceil
from typing import Final


PCM_CHANNELS: Final = 1
PCM_SAMPLE_WIDTH_BYTES: Final = 2


def milliseconds_to_frames(
    duration_ms: int,
    *,
    sample_rate: int,
    frame_samples: int,
) -> int:
    return max(1, ceil(duration_ms * sample_rate / (1_000 * frame_samples)))
