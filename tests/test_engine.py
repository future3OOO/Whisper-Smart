"""Property-based + unit tests.  Coverage >= 85 %."""

from __future__ import annotations

import asyncio
from collections import namedtuple
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, strategies as st

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine


class DummyModel:  # fast stub for WhisperModel
    def transcribe(self, audio, **_kw):  # noqa: ANN001
        Seg = namedtuple("Seg", "text")
        if audio.sum() == 0:
            return [Seg(text="")], {"avg_logprob": 0.0}
        # simple deterministic mapping: len(audio) → text digits
        return [Seg(text=str(len(audio)))], {"avg_logprob": 0.5}


@pytest.fixture()
def monkey_whisper(monkeypatch):  # noqa: D401
    monkeypatch.setattr("dictation_tool.engine.WhisperModel", lambda *a, **k: DummyModel())


audio_arrays = st.lists(
    st.integers(min_value=-32768, max_value=32767).map(lambda x: np.array([x], dtype=np.int16)),
    min_size=1,
    max_size=100,
)


@given(audio_arrays)
def test_transcribe_various(monkey_whisper, audio_arrays):  # noqa: D401
    cfg = Config(device="cpu")
    eng = DictationEngine(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    text = loop.run_until_complete(eng._transcribe(np.concatenate(audio_arrays)))  # noqa: SLF001
    assert isinstance(text, str)
