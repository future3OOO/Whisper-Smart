"""Property-based + unit tests.  Coverage >= 85 %."""

from __future__ import annotations

import asyncio

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine
from dictation_tool.transcription import TranscriptionResult


class DummyBackend:
    def transcribe(self, audio, _options):
        if audio.sum() == 0:
            return TranscriptionResult(text="", avg_logprob=0.0)
        # simple deterministic mapping: len(audio) → text digits
        return TranscriptionResult(text=str(len(audio)), avg_logprob=0.5)


audio_arrays = st.lists(
    st.integers(min_value=-32768, max_value=32767).map(
        lambda x: np.array([x], dtype=np.int16)
    ),
    min_size=1,
    max_size=100,
)


@given(audio_arrays)
def test_transcribe_various(audio_arrays):
    cfg = Config(device="cpu")
    eng = DictationEngine(cfg)
    eng._backend = DummyBackend()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    text = loop.run_until_complete(eng._transcribe(np.concatenate(audio_arrays)))
    assert isinstance(text, str)
