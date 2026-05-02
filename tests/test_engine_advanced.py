"""Advanced tests for the optimised DictationEngine."""

from unittest.mock import ANY, AsyncMock, Mock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dictation_tool import engine as engine_module
from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine, _Ring
from dictation_tool.transcription import TranscriptionResult

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
    def mock_backend(self):
        """A stub transcription backend."""
        backend = Mock()

        def mock_transcribe(_audio, options):
            temp = options.temperature
            if temp == 0.0:
                return TranscriptionResult(text="", avg_logprob=-1.0)
            return TranscriptionResult(text="hello world", avg_logprob=-0.1)

        backend.transcribe = mock_transcribe
        return backend

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
        assert eng._backend is None
        assert eng._ring is None
        assert not eng._raw_shadow

    # ── model-load path (GPU) ───────────────────────────────────
    @patch("dictation_tool.engine.os.cpu_count", return_value=24)
    @patch("dictation_tool.engine.FasterWhisperBackend")
    @pytest.mark.asyncio
    async def test_load_model_flash(self, mock_backend_cls, _mock_cpu_count):
        backend = Mock()
        mock_backend_cls.return_value = backend
        cfg = Config(device="cuda", attention_backend="flash")
        eng = DictationEngine(cfg)

        await eng._load_model()

        mock_backend_cls.assert_called_once_with(
            model_name="large-v3-turbo",
            device="cuda",
            compute_type="float16",
            cpu_threads=12,
            num_workers=12,
        )
        backend.load.assert_called_once()
        assert eng._backend is backend

    # ── model-load path (CPU) ───────────────────────────────────
    @patch("dictation_tool.engine.FasterWhisperBackend")
    @pytest.mark.asyncio
    async def test_load_model_cpu_no_compile(self, mock_backend_cls):
        mock_backend_cls.return_value = Mock()
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)

        with patch("dictation_tool.engine.torch.compile") as m_compile:
            await eng._load_model()
            m_compile.assert_not_called()

    # ── trigger install ────────────────────────────────────────
    @patch("dictation_tool.engine.add_hotkey")
    @patch("dictation_tool.engine.mouse.Listener")
    def test_install_triggers(self, m_listener, m_hotkey):
        cfg = Config(
            hotkey="ctrl+space",
            mouse_btn="middle",
            enable_mouse_trigger=True,
        )
        DictationEngine(cfg)._install_triggers()
        m_hotkey.assert_called_once_with("ctrl+space", ANY)
        m_listener.assert_called_once()

    # ── transcription logic ─────────────────────────────────────
    @pytest.mark.asyncio
    async def test_transcription_returns_model_text(self, mock_backend):
        cfg = Config(device="cpu", attention_backend="none")

        mock_backend.transcribe = Mock(
            return_value=TranscriptionResult(text="hello world", avg_logprob=-0.1)
        )
        eng = DictationEngine(cfg)
        eng._backend = mock_backend
        audio = np.ones(16000, dtype=np.int16) * 1000
        assert await eng._transcribe(audio) == "Hello world"

    @pytest.mark.asyncio
    async def test_transcription_early_exit(self, mock_backend):
        cfg = Config(device="cpu", attention_backend="none")

        mock_backend.transcribe = Mock(
            return_value=TranscriptionResult(text="ok", avg_logprob=-0.1)
        )
        eng = DictationEngine(cfg)
        eng._backend = mock_backend
        audio = np.ones(8000, dtype=np.int16) * 1000
        assert await eng._transcribe(audio) == "Ok"

    @pytest.mark.asyncio
    async def test_benchmark_rtf_uses_reported_inference_latency(self, monkeypatch):
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)
        eng._backend = Mock()
        eng._transcribe = AsyncMock(return_value="hello")
        times = iter([0.0, 0.0, 0.0, 0.1, 0.1, 0.3])
        monkeypatch.setattr(engine_module.time, "perf_counter", lambda: next(times))

        result = await eng.run_benchmark(audio=np.ones(16000, dtype=np.int16), runs=2)

        assert result["inference_ms"] == 200.0
        assert result["real_time_factor"] == 5.0

    # ── flush path & clipboard ─────────────────────────────────
    @patch("dictation_tool.engine._paste_retry")
    @pytest.mark.asyncio
    async def test_clip_worker_copies_and_pastes(self, m_paste):
        cfg = Config(
            device="cpu",
            attention_backend="none",
            auto_paste_on_release=True,
        )
        eng = DictationEngine(cfg)
        eng._clip.copy = AsyncMock()
        await eng._clip_q.put("hello")
        await eng._clip_q.put(None)

        await eng._clip_worker()

        eng._clip.copy.assert_called_once_with("hello")
        m_paste.assert_called_once()

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

    @pytest.mark.asyncio
    async def test_flush_hold_does_not_duplicate_vad_tail(self):
        cfg = Config(device="cpu", attention_backend="none", use_vad=True)
        eng = DictationEngine(cfg)
        shadow = np.ones(1600, dtype=np.int16)
        tail = np.ones(800, dtype=np.int16)
        captured = {}

        async def transcribe(audio):
            captured["samples"] = audio.size
            return "hello"

        eng._raw_shadow.append(shadow)
        eng._vad_gate = Mock(force_flush=Mock(return_value=tail))
        eng._transcribe = AsyncMock(side_effect=transcribe)

        assert await eng._flush_hold() == "hello"
        assert captured["samples"] == shadow.size
        eng._vad_gate.force_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_hold_snapshots_shadow_before_concatenate(self, monkeypatch):
        cfg = Config(device="cpu", attention_backend="none", use_vad=True)
        eng = DictationEngine(cfg)
        shadow = np.ones(1600, dtype=np.int16)
        concurrent = np.full(800, 2, dtype=np.int16)
        captured = {}
        added = False
        original_concatenate = engine_module.concatenate

        def interleaving_concatenate(chunks):
            nonlocal added
            if not added:
                added = True
                eng._add_to_shadow(concurrent)
            return original_concatenate(chunks)

        async def transcribe(audio):
            captured["samples"] = audio.size
            return "hello"

        monkeypatch.setattr(engine_module, "concatenate", interleaving_concatenate)
        eng._add_to_shadow(shadow)
        eng._transcribe = AsyncMock(side_effect=transcribe)

        assert await eng._flush_hold() == "hello"
        assert captured["samples"] == shadow.size
        assert len(eng._raw_shadow) == 1
        np.testing.assert_array_equal(eng._raw_shadow[0], concurrent)

    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_hold_mode_does_not_transcribe_vad_batches_before_release(
        self, m_stream
    ):
        cfg = Config(device="cpu", attention_backend="none", use_vad=True)
        eng = DictationEngine(cfg)
        eng._holding = True
        eng._recording.set()
        eng._transcribe = AsyncMock(return_value="should not paste")

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            eng._terminate.set()
            yield np.ones(1600, dtype=np.int16)

        stub.chunks.return_value = gen()
        m_stream.return_value = stub

        await eng._run()
        eng._transcribe.assert_not_called()

    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_hold_mode_clears_stale_vad_batch_before_resume(self, m_stream):
        cfg = Config(device="cpu", attention_backend="none", use_vad=True)
        eng = DictationEngine(cfg)
        eng._recording.set()
        eng._batch_ctl.min_samples = 4
        eng._batch_ctl.max_chunks = 3
        captured = []

        async def transcribe(audio):
            captured.append(audio.copy())
            return "after hold"

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            yield np.array([1, 1], dtype=np.int16)
            eng._holding = True
            yield np.array([2, 2], dtype=np.int16)
            eng._holding = False
            yield np.array([3, 3], dtype=np.int16)
            eng._terminate.set()

        stub.chunks.return_value = gen()
        m_stream.return_value = stub
        eng._transcribe = AsyncMock(side_effect=transcribe)

        await eng._run()
        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0], np.array([3, 3], dtype=np.int16))

    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_run_without_vad(self, m_stream):
        cfg = Config(device="cpu", attention_backend="none", use_vad=False)
        eng = DictationEngine(cfg)
        eng._recording.set()

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
        assert m_stream.call_args.kwargs["on_raw_chunk"].__self__ is eng

    # ── buffer overflow branch ─────────────────────────────────
    def test_ring_pop_returns_full_buffer(self):
        ring = _Ring(4)

        ring.push(np.array([1, 2, 3, 4], dtype=np.int16))

        assert ring.size == 4
        np.testing.assert_array_equal(
            ring.pop(),
            np.array([1, 2, 3, 4], dtype=np.int16),
        )

    def test_ring_exact_fill_after_wrap_returns_all_samples(self):
        ring = _Ring(8)

        ring.push(np.arange(6, dtype=np.int16))
        ring.pop()
        ring.push(np.arange(10, 17, dtype=np.int16))
        ring.push(np.array([17], dtype=np.int16))

        assert ring.size == 8
        np.testing.assert_array_equal(ring.pop(), np.arange(10, 18, dtype=np.int16))

    def test_ring_wrap_overwrite_keeps_latest_capacity(self):
        ring = _Ring(8)

        ring.push(np.arange(6, dtype=np.int16))
        ring.push(np.arange(6, 10, dtype=np.int16))

        assert ring.size == 8
        np.testing.assert_array_equal(ring.pop(), np.arange(2, 10, dtype=np.int16))

    @patch("dictation_tool.engine.AudioStream")
    @pytest.mark.asyncio
    async def test_buffer_overflow(self, m_stream):
        cfg = Config(
            device="cpu",
            attention_backend="none",
            use_vad=False,
            max_buffer_seconds=6,
            sample_rate=16000,
        )
        eng = DictationEngine(cfg)
        eng._transcribe = AsyncMock(return_value="hi")

        stub = Mock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=None)

        async def gen():
            eng._recording.set()
            assert eng._ring is not None
            eng._ring.push(np.ones(100000, dtype=np.int16))
            yield np.empty(0, dtype=np.int16)
            eng._terminate.set()

        stub.chunks.return_value = gen()
        m_stream.return_value = stub

        await eng._run()
        eng._transcribe.assert_called_once()

    # ── property-based sanity on counters ──────────────────────
    @given(st.lists(st.integers(min_value=0, max_value=32767), min_size=1, max_size=50))
    def test_chunk_counter(self, vals):
        cfg = Config(device="cpu", attention_backend="none")
        eng = DictationEngine(cfg)
        chunk = np.array(vals, dtype=np.int16)
        eng._add_to_shadow(chunk)
        np.testing.assert_array_equal(eng._raw_shadow[0], chunk)

    # ── misc integration ───────────────────────────────────────
    def test_stop_sets_flag(self):
        eng = DictationEngine(Config(device="cpu", attention_backend="none"))
        eng.stop()
        assert eng._terminate.is_set()

    @pytest.mark.asyncio
    async def test_run_benchmark_reports_latency_metrics(self):
        cfg = Config(device="cpu", attention_backend="none", sample_rate=16000)
        eng = DictationEngine(cfg)
        eng._load_model = AsyncMock()
        eng._transcribe = AsyncMock(return_value="benchmark transcript")

        result = await eng.run_benchmark(seconds=0.25)

        assert result["audio_seconds"] == 0.25
        assert result["samples"] == 4000
        assert result["model_load_ms"] >= 0
        assert result["inference_ms"] >= 0
        assert result["real_time_factor"] >= 0
        eng._load_model.assert_called_once()
        eng._transcribe.assert_called_once()
