"""Transcription backend adapters."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from numpy.typing import NDArray


@dataclass(frozen=True)
class TranscriptionOptions:
    language: str | None = None
    initial_prompt: str | None = None
    beam_size: int = 5
    best_of: int = 1
    temperature: float = 0.0
    vad_filter: bool = True


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    avg_logprob: float


class FasterWhisperBackend:
    """Small adapter around faster-whisper model loading and inference."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 1,
        num_workers: int = 1,
        model_factory: Callable[..., Any] = WhisperModel,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._model_factory: Callable[..., Any] = model_factory
        self._model: Any | None = None

    def load(self) -> None:
        self._model = self._model_factory(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
            num_workers=self._num_workers,
        )

    def close(self) -> None:
        self._model = None
        gc.collect()

    def transcribe(
        self, audio: NDArray[np.float32], options: TranscriptionOptions
    ) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("transcription backend is not loaded")

        segs, _info = self._model.transcribe(
            audio,
            language=options.language,
            initial_prompt=options.initial_prompt,
            beam_size=options.beam_size,
            best_of=options.best_of,
            temperature=options.temperature,
            vad_filter=options.vad_filter,
            word_timestamps=False,
        )
        segments = list(segs)
        return TranscriptionResult(
            text="".join(segment.text for segment in segments).strip(),
            avg_logprob=float(
                sum(segment.avg_logprob for segment in segments) / len(segments)
                if segments
                else -1.0
            ),
        )
