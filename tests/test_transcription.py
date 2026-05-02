"""Tests for transcription backend adapters."""

from collections import namedtuple

import numpy as np
import pytest

from dictation_tool.transcription import FasterWhisperBackend, TranscriptionOptions


def test_faster_whisper_backend_loads_model_with_runtime_config():
    calls = []

    class Model:
        pass

    def model_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return Model()

    backend = FasterWhisperBackend(
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        cpu_threads=6,
        num_workers=3,
        model_factory=model_factory,
    )

    backend.load()

    assert calls == [
        (
            ("large-v3-turbo",),
            {
                "device": "cuda",
                "compute_type": "float16",
                "cpu_threads": 6,
                "num_workers": 3,
            },
        )
    ]


def test_faster_whisper_backend_transcribes_with_options():
    seen = {}
    segment = namedtuple("Segment", "text avg_logprob")

    class Model:
        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            seen["kwargs"] = kwargs
            return [
                segment("hello ", -0.2),
                segment("world", -0.4),
            ], object()

    backend = FasterWhisperBackend(
        model_name="large-v3",
        device="cpu",
        compute_type="int8",
        model_factory=lambda *args, **kwargs: Model(),
    )
    backend.load()
    audio = np.ones(16000, dtype=np.float32)

    result = backend.transcribe(
        audio,
        TranscriptionOptions(
            language="en",
            initial_prompt="Prompt",
            beam_size=5,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
        ),
    )

    assert result.text == "hello world"
    assert result.avg_logprob == pytest.approx(-0.3)
    np.testing.assert_array_equal(seen["audio"], audio)
    assert seen["kwargs"] == {
        "language": "en",
        "initial_prompt": "Prompt",
        "beam_size": 5,
        "best_of": 1,
        "temperature": 0.0,
        "vad_filter": True,
        "word_timestamps": False,
    }


def test_faster_whisper_backend_fails_closed_before_load():
    backend = FasterWhisperBackend(
        model_name="large-v3",
        device="cpu",
        compute_type="int8",
        model_factory=lambda *args, **kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="not loaded"):
        backend.transcribe(
            np.ones(16000, dtype=np.float32),
            TranscriptionOptions(),
        )
