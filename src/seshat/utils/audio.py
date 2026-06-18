from __future__ import annotations

import io
import logging
import math

import mutagen

logger = logging.getLogger(__name__)


def audio_duration_seconds(audio_bytes: bytes) -> float | None:
    """Return audio duration in seconds, or None if mutagen cannot parse the format."""
    try:
        audio_file = mutagen.File(io.BytesIO(audio_bytes))
    except Exception as exc:
        logger.warning("Failed to parse audio bytes for duration: %s: %s", type(exc).__name__, exc)
        return None

    if audio_file is None or not hasattr(audio_file.info, "length"):
        logger.warning("Failed to parse audio bytes for duration: format not recognised")
        return None
    return audio_file.info.length


def audio_duration_seconds_ceil(audio_bytes: bytes) -> int | None:
    """Return audio duration rounded up to the nearest second, or None if unparseable."""
    duration = audio_duration_seconds(audio_bytes)
    if duration is None:
        return None
    return math.ceil(duration)
