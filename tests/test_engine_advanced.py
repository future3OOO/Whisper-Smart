"""Advanced tests for the optimised DictationEngine."""

from collections import namedtuple
from unittest.mock import ANY, AsyncMock, Mock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine

# ────────────────────────────────────────────────────────────────
# Hypothesis: disable deadline to keep CI stable
settings.register_profile(
    "ci",
    deadline=None,
)
settings.load_profile("ci")


class TestAdvancedDictationEngine:
    """Test suite for DictationEngine internal behaviours."""

    # ── fixtures ────────────────────────────────────────────────
    @pytest.fixture
    def mock_whisper_model(self):
        """A stub WhisperModel with adaptive-temp behaviour."""
        mock_model = Mock()
        mock_model.model = Mock()  # torch.compile expects .model attr

        def mock_transcribe(_audio, **kw):
            temp = kw.get("temperature", 0.0)
            Seg = namedtuple("Seg", "text")
            Info = namedtuple("Info", "no_speech_prob")
            if temp == 0.0:
                return [Seg(text="")], Info(no_speech_prob=0.8)
            return [Seg(text="hello world")], Info(no_speech_prob=0.1)

        mock_model.transcribe = mock_transcribe
        return mock_model

    # ── basic object state ──────────────────────────────────────
    def test_engine_initialisation(self):
        cfg = Config(
            device="cuda",
            attention_backend="flash",
            use_vad=True,
            vad_aggr=3,
            mouse_btn="right",
        )
        eng = DictationEngine(cfg)
        assert eng.cfg == cfg
        assert eng._model is None
        assert eng._buff_samples == 0
        assert not eng._chunks

    # ── model-load path (GPU) ───────────────────────────────────
    @patch("dictation_tool.engine.torch.compile")
    @patch("dictation_tool.engine.WhisperModel")
    @pytest.mark.asyncio
    async def test_load_model_flash(self, mock_whisper_cls, mock_torch_compile):
        mock_model = Mock(model=Mock())
        mock_whisper_cls.return_value = mock_model
        original_model = mock_model.model

        cfg = Config(device="cuda", attention_backend="flash")
        eng = DictationEngine(cfg)

        await eng._load_model()

        mock_whisper_cls.assert_called_once_with("large-v3", device="cuda", compute_type="float16")
        mock_torch_compile.assert_called_once()
        # Check that torch.compile was called with the original model
        assert mock_torch_compile.call_args.args[0] is original_model

    # ── model-load path (CPU) ───────────────────────────────────
    @patch("dictation_tool.engine.WhisperModel")
    @pytest.mark.asyncio
    async def test_load_model_cpu_no_compile(self, mock_whisper_cls):
        mock_whisper_cls.return_value = Mock()
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)

        with patch("dictation_tool.engine.torch.compile") as m_compile:
            await eng._load_model()
            m_compile.assert_not_called()

    # ── trigger install ────────────────────────────────────────
    @patch("dictation_tool.engine.add_hotkey")
    @patch("dictation_tool.engine.mouse.Listener")
    def test_install_triggers(self, m_listener, m_hotkey):
        cfg = Config(hotkey="ctrl+space", mouse_btn="middle")
        DictationEngine(cfg)._install_triggers()
        m_hotkey.assert_called_once_with("ctrl+space", ANY)
        m_listener.assert_called_once()

    # ── adaptive transcription logic ────────────────────────────
    @pytest.mark.asyncio
    async def test_adaptive_retry(self, mock_whisper_model):
        cfg = Config(device="cpu", attention_backend="none", retry_temperatures=(0.0, 0.4))
        eng = DictationEngine(cfg)
        eng._model = mock_whisper_model
        text = await eng._transcribe(np.zeros(16000, dtype=np.int16))
        assert text == "hello world"

    @pytest.mark.asyncio
    async def test_transcription_early_exit(self, mock_whisper_model):
        cfg = Config(device="cpu", attention_backend="none")

        def good_transcribe(_a, **_kw):
            Seg = namedtuple("Seg", "text")
            Info = namedtuple("Info", "no_speech_prob")
            return [Seg(text="ok")], Info(no_speech_prob=0.1)

        mock_whisper_model.transcribe = good_transcribe
        eng = DictationEngine(cfg)
        eng._model = mock_whisper_model
        assert await eng._transcribe(np.zeros(8000, dtype=np.int16)) == "ok"

    # ── flush path & clipboard ─────────────────────────────────
    @patch("dictation_tool.engine.retry")
    @pytest.mark.asyncio
    async def test_flush_copies_clipboard(self, m_retry, mock_whisper_model):
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)
        eng._model = mock_whisper_model
        eng._chunks.extend(
            [
                np.array([1, 2, 3], dtype=np.int16),
                np.array([4, 5, 6], dtype=np.int16),
            ]
        )
        eng._buff_samples = 6
        await eng._flush()
        m_retry.assert_called_once()
        assert not eng._chunks and eng._buff_samples == 0

    # ── run loop: VAD vs no-VAD branches ───────────────────────
    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_run_with_vad(self, m_stream):
        cfg = Config(device="cpu", attention_backend="none", use_vad=True)
        eng = DictationEngine(cfg)

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            yield np.array([1, 2], dtype=np.int16)
            eng._terminate.set()

        stub.chunks.return_value = gen()
        m_stream.return_value = stub

        await eng._run()
        assert m_stream.call_args.kwargs["vad_gate"] is not None

    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_run_without_vad(self, m_stream):
        cfg = Config(device="cpu", attention_backend="none", use_vad=False)
        eng = DictationEngine(cfg)

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            yield np.array([1, 2], dtype=np.int16)
            eng._terminate.set()

        stub.chunks.return_value = gen()
        m_stream.return_value = stub

        await eng._run()
        assert m_stream.call_args.kwargs["vad_gate"] is None

    # ── buffer overflow branch ─────────────────────────────────
    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_buffer_overflow(self, m_stream):
        cfg = Config(
            device="cpu",
            attention_backend="none",
            max_buffer_seconds=0.001,
            sample_rate=16000,
        )
        eng = DictationEngine(cfg)
        eng._model = Mock(
            transcribe=lambda *_a, **_k: (
                [namedtuple("Seg", "text")(text="hi")],
                namedtuple("Info", "no_speech_prob")(no_speech_prob=0.1),
            )
        )

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            eng._recording.set()
            yield np.array([1] * 1000, dtype=np.int16)
            eng._terminate.set()

        stub.chunks.return_value = gen()
        m_stream.return_value = stub

        with patch("dictation_tool.engine.retry"):
            await eng._run()

    # ── property-based sanity on counters ──────────────────────
    @given(st.lists(st.integers(min_value=0, max_value=32767), min_size=1, max_size=50))
    def test_chunk_counter(self, vals):
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)
        chunk = np.array(vals, dtype=np.int16)
        eng._chunks.append(chunk)
        eng._buff_samples += len(chunk)
        assert eng._buff_samples == len(vals)

    # ── misc integration ───────────────────────────────────────
    def test_stop_sets_flag(self):
        eng = DictationEngine(Config(device="cpu", attention_backend="none"))
        eng.stop()
        assert eng._terminate.is_set()
