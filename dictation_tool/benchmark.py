"""Benchmark and accuracy helpers."""

from __future__ import annotations

import re
import wave
from collections.abc import Iterable
from math import ceil
from pathlib import Path
from statistics import median

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_MATRIX = (
    "large-v3",
    "medium.en",
    "large-v3-turbo",
    "distil-large-v3",
)

Int16Audio = NDArray[np.int16]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def parse_model_matrix(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated model list, preserving order without duplicates."""
    if value is None or value.strip().lower() == "default":
        return DEFAULT_MODEL_MATRIX

    models: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        model = raw.strip()
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    if not models:
        raise ValueError("at least one benchmark model is required")
    return tuple(models)


def summarize_latency_ms(values: Iterable[float]) -> dict[str, float | int]:
    """Return stable latency summary metrics for benchmark samples."""
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("at least one latency sample is required")

    p95_index = ceil(0.95 * len(samples)) - 1
    return {
        "count": len(samples),
        "min_ms": round(samples[0], 3),
        "p50_ms": round(float(median(samples)), 3),
        "p95_ms": round(samples[p95_index], 3),
        "max_ms": round(samples[-1], 3),
    }


def load_wav_mono_int16(path: Path, sample_rate: int) -> Int16Audio:
    """Load a PCM WAV fixture and resample to the engine sample rate if needed."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        source_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("benchmark WAV fixtures must be 16-bit PCM")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if source_rate == sample_rate:
        return audio.copy()

    target_frames = max(1, int(audio.size * sample_rate / source_rate))
    source_idx = np.arange(audio.size, dtype=np.float64)
    target_idx = np.linspace(0, audio.size - 1, target_frames, dtype=np.float64)
    return np.interp(target_idx, source_idx, audio.astype(np.float64)).astype(np.int16)


def word_error_rate(reference: str, candidate: str) -> float:
    """Compute word error rate with a small Levenshtein distance implementation."""
    ref = TOKEN_RE.findall(reference.lower())
    hyp = TOKEN_RE.findall(candidate.lower())
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1] / len(ref)
