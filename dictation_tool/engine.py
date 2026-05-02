"""
dictation_tool.engine - v3.3-titan (2025-06-08)

Key features
────────────
• Unlimited shadow buffer during mouse-hold (guarantees full 60 s capture)
• Soft RMS gate (8 000) - keeps very quiet consonants
• Thread-pool clipboard copy, fast Win32 path + retry paste
• Flash-SDP / TF-32 GPU tuning
• Regex post-processor:
    "foo at bar dot com"  →  foo@bar.com
    "www . example . com"→  www.example.com
• Extended punctuation map (“at sign” → @)
• JSONL profiler, adaptive batching, back-pressure, dynamic ring growth
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil

import numpy as np
import pyperclip  # type: ignore[import-untyped]
import torch
from keyboard import add_hotkey, is_pressed  # type: ignore[import-untyped]
from keyboard import send as kb_send
from numpy.typing import NDArray
from pynput import mouse  # type: ignore[import-untyped]

from .benchmark import summarize_latency_ms, word_error_rate
from .config import Config
from .io import AudioStream, Int16Audio, VADGate, concatenate
from .postprocess import DictationPostProcessor
from .transcription import FasterWhisperBackend, TranscriptionOptions
from .utils import LOGGER, timed
from .utils.profile import prof

Float32Audio = NDArray[np.float32]
Int32Audio = NDArray[np.int32]

# ══════════════════════════════════ Clipboard helpers ═════════════════════════
if sys.platform == "win32":
    try:
        import win32clipboard as _wc  # type: ignore
        import win32con as _wcon  # type: ignore
    except ImportError:  # pywin32 not installed
        _wc = _wcon = None
else:
    _wc = _wcon = None


def _fast_win_clip(text: str) -> None:
    """~45 µs RTT clipboard writer (Windows only)."""
    if not _wc:
        raise RuntimeError("pywin32 unavailable")
    _wc.OpenClipboard()
    try:
        _wc.EmptyClipboard()
        _wc.SetClipboardData(_wcon.CF_UNICODETEXT, text)
    finally:
        _wc.CloseClipboard()


def _paste_retry() -> None:
    """Ctrl/Cmd-V with up to three attempts (20 ms back-off)."""
    for attempt in range(3):
        try:
            if sys.platform == "win32":
                user32 = ctypes.windll.user32
                CTRL, V, KEYUP = 0x11, 0x56, 0x0002
                user32.keybd_event(CTRL, 0, 0, 0)
                user32.keybd_event(V, 0, 0, 0)
                user32.keybd_event(V, 0, KEYUP, 0)
                user32.keybd_event(CTRL, 0, KEYUP, 0)
            else:
                kb_send("command+v" if sys.platform == "darwin" else "ctrl+v")
            return
        except Exception as exc:
            LOGGER.debug("Paste retry %d/3 failed: %s", attempt + 1, exc)
            time.sleep(0.02)
    LOGGER.warning("Auto-paste ultimately failed")


# ══════════════════════════════════ Audio ring buffer ═════════════════════════
class _Ring:
    """Lock-free power-of-two ring for int16 audio."""

    __slots__ = ("_buf", "_full", "_head", "_mask", "_tail", "_view")

    def __init__(self, cap: int) -> None:
        cap = 1 << (cap - 1).bit_length()
        self._buf: Int16Audio = np.zeros(cap, np.int16)
        self._mask = cap - 1
        self._head = self._tail = 0
        self._full = False
        self._view = self._buf

    def push(self, chunk: Int16Audio) -> None:
        if chunk.ndim > 1:
            chunk = chunk.ravel()
        n = int(chunk.size)
        if not n:
            return
        cap = self._view.shape[0]
        if n >= cap:  # keep only the last <cap> samples
            self._view[:] = chunk[-cap:]
            self._head = self._tail = 0
            self._full = True
            return
        size = self.size
        head, end = self._head, self._head + n
        if end <= cap:
            self._view[head:end] = chunk
        else:
            cut = cap - head
            self._view[head:] = chunk[:cut]
            self._view[: end - cap] = chunk[cut:]
        self._head = end & self._mask
        if self._full or size + n >= cap:
            self._tail = self._head
            self._full = True

    def pop(self) -> Int16Audio:
        if self._head == self._tail and not self._full:
            return np.empty(0, np.int16)
        if self._full:
            out = (
                self._buf.copy()
                if self._head == 0
                else np.concatenate((self._buf[self._tail :], self._buf[: self._head]))
            )
        elif self._head > self._tail:
            out = self._buf[self._tail : self._head].copy()
        else:
            out = np.concatenate((self._buf[self._tail :], self._buf[: self._head]))
        self._tail = self._head
        self._full = False
        return out

    @property
    def size(self) -> int:
        return (
            self._view.shape[0]
            if self._full
            else (self._head - self._tail) & self._mask
        )


# ═════════════════════════ Thread-safe context ════════════════════════════════
class _Context:
    """Rolling prompt history (mutex-guarded)."""

    def __init__(self, hist: int = 5) -> None:
        self._buf: deque[str] = deque(maxlen=hist)
        self._lock = threading.Lock()

    def push(self, txt: str) -> None:
        if len(txt.split()) < 3:
            return
        with self._lock:
            self._buf.append(txt.strip())

    def prompt(self) -> str:
        with self._lock:
            return " ".join(self._buf) + ". " if self._buf else ""


# ═══════════════════ Adaptive batch controller ════════════════════════════════
class _BatchCtl:
    def __init__(self, cfg: Config) -> None:
        self.min_samples = cfg.batch_min_samples
        self.max_chunks = cfg.batch_max_chunks
        self._lock = not cfg.adaptive_batching
        self._seen = self._chunks = 0

    def feed(self, samples: int) -> None:
        if self._lock:
            return
        self._seen += samples
        self._chunks += 1
        if self._seen >= 32_000:  # ~2 s @16 kHz
            avg = self._seen // self._chunks
            self.min_samples = max(8_000, min(int(avg * 0.8) // 16 * 16, 24_000))
            self.max_chunks = max(4, min(ceil(self.min_samples / avg), 10))
            LOGGER.info(
                "Adaptive batch tuned → %d samples | %d chunks",
                self.min_samples,
                self.max_chunks,
            )
            self._lock = True


# ═════════════════════ Clipboard wrapper class ════════════════════════════════
class _Clipboard:
    """Thread-pool clipboard copy; never blocks event loop."""

    def __init__(self) -> None:
        self._last = ("", 0.0)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def _sync_copy(self, text: str) -> None:
        if sys.platform == "win32" and _wc:
            _fast_win_clip(text)
        else:
            pyperclip.copy(text)

    async def copy(self, text: str) -> None:
        txt, ts = self._last
        if text == txt and (time.time() - ts) < 0.1:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._pool, self._sync_copy, text)
        except Exception as exc:
            LOGGER.warning("Clipboard fast-path failed: %s - fallback", exc)
            await loop.run_in_executor(self._pool, pyperclip.copy, text)
        self._last = (text, time.time())


# ══════════════════════════ Dictation Engine ══════════════════════════════════
class DictationEngine:
    """Microphone → (VAD) → Whisper → clipboard."""

    # ───────────────────────── init ──────────────────────────
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cap = int(cfg.sample_rate * cfg.max_buffer_seconds * 2)
        self._f32: Float32Audio = np.empty(cap, np.float32)
        self._i32: Int32Audio = np.empty(cap, np.int32)
        self._ring = _Ring(cap) if not cfg.use_vad else None

        self._backend: FasterWhisperBackend | None = None
        self._model_pool = ThreadPoolExecutor(max_workers=1)
        self._recording = asyncio.Event()
        self._terminate = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_lock = threading.Lock()

        self._clip = _Clipboard()
        self._clip_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=10)
        self._post = DictationPostProcessor()
        self._ctx = _Context()
        self._batch_ctl = _BatchCtl(cfg)

        # triggers
        self._mouse_listener: mouse.Listener | None = None
        self._mouse_press: float | None = None
        self._hold_timer: threading.Timer | None = None
        self._holding = False

        # shadow buffer
        self._raw_shadow: deque[Int16Audio] = (
            deque()
            if cfg.mouse_hold_to_record
            else deque(maxlen=int(cfg.max_buffer_seconds * 1000 / cfg.chunk_ms))
        )
        self._vad_gate: VADGate | None = None

        self._rms: dict[tuple[int, int, int, int], int] = {}

    # ──────────────────── GPU tuning ─────────────────────────
    def _tune_cuda(self) -> None:
        if self.cfg.device != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            maj, min_ = torch.cuda.get_device_capability()
            tmaj, tmin = map(int, torch.__version__.split("+")[0].split(".")[:2])
            if (maj, min_) >= (8, 0) and (tmaj, tmin) >= (2, 0):
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)
                    LOGGER.info("Flash-Attention enabled")
        except Exception as exc:
            LOGGER.debug("CUDA tuning skipped: %s", exc)

    # ──────────────────── model load ─────────────────────────
    async def _load_model(self) -> None:
        if self.cfg.compute_type == "auto":
            ct = "float16" if self.cfg.device == "cuda" else "float32"
            object.__setattr__(self.cfg, "compute_type", ct)

        def _sync() -> FasterWhisperBackend:
            th = max(1, (os.cpu_count() or 4) // 2)
            backend = FasterWhisperBackend(
                model_name=self.cfg.model_name,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=th,
                num_workers=th,
            )
            backend.load()
            return backend

        LOGGER.info("Loading %s …", self.cfg.model_name)
        profile = str(self.cfg.profile) if self.cfg.profile else None
        with prof("model_load", profile), timed("model-load"):
            self._backend = await asyncio.get_running_loop().run_in_executor(
                self._model_pool, _sync
            )

        self._tune_cuda()
        for n in (8_000, 16_000):
            dummy = np.random.randint(-500, 500, n, np.int16)
            try:
                await self._transcribe(dummy)
            except Exception:
                pass
        LOGGER.info("Model ready ✔")

    # ──────────────────── clipboard worker ───────────────────
    async def _clip_worker(self) -> None:
        while True:
            txt = await self._clip_q.get()
            if txt is None:
                break
            await self._clip.copy(txt)
            if self.cfg.auto_paste_on_release:
                await asyncio.sleep(0.01)
                _paste_retry()
            LOGGER.debug("📋 Sent")

    # ──────────────────── transcription ──────────────────────
    async def _transcribe(self, audio: Int16Audio) -> str:
        if audio.ndim > 1:
            audio = audio.squeeze()
        n = int(audio.size)
        if n < 800:
            return ""

        # soft RMS gate
        if n < 4_096:
            energy = int(np.square(audio, dtype=np.int32).sum(dtype=np.int64) // n)
            if energy < 8_000:
                return ""
        else:
            k = (n, int(audio[0]), int(audio[n // 2]), int(audio[-1]))
            if k not in self._rms:
                if n > self._i32.size:
                    self._i32 = np.empty(int(n * 1.3), np.int32)
                np.square(audio, out=self._i32[:n], dtype=np.int32)
                self._rms[k] = int(self._i32[:n].sum(dtype=np.int64) // n)
                if len(self._rms) > 64:
                    self._rms.clear()
            if self._rms[k] < 8_000:
                return ""

        if n > self._f32.size:
            self._f32 = np.empty(int(n * 1.3), np.float32)
        np.multiply(audio, 1 / 32768.0, out=self._f32[:n], dtype=np.float32)
        if self.cfg.mic_gain != 1.0:
            self._f32[:n] *= self.cfg.mic_gain
        np.clip(self._f32[:n], -1.0, 1.0, out=self._f32[:n])
        if self._backend is None:
            raise RuntimeError("transcription backend is not loaded")

        profile = str(self.cfg.profile) if self.cfg.profile else None
        inference_audio = self._f32[:n].copy()
        with prof("infer", profile), timed("infer"):
            result = await asyncio.get_running_loop().run_in_executor(
                self._model_pool,
                self._backend.transcribe,
                inference_audio,
                TranscriptionOptions(
                    language=self.cfg.language,
                    initial_prompt=self.cfg.initial_prompt or self._ctx.prompt(),
                    beam_size=self.cfg.beam_size,
                    best_of=self.cfg.best_of,
                    temperature=self.cfg.temperature,
                    vad_filter=self.cfg.model_vad_filter,
                ),
            )

        txt = result.text
        if txt and result.avg_logprob > -0.6:
            self._ctx.push(txt)

        return self._post.clean_model_text(txt)

    # ──────────────────── shadow helper ──────────────────────
    def _add_to_shadow(self, chunk: Int16Audio) -> None:
        with self._audio_lock:
            self._raw_shadow.append(chunk)

    def _pop_raw_shadow(self) -> list[Int16Audio]:
        with self._audio_lock:
            chunks = list(self._raw_shadow)
            self._raw_shadow.clear()
        return chunks

    def _clear_raw_shadow(self) -> None:
        with self._audio_lock:
            self._raw_shadow.clear()

    def _add_to_ring(self, chunk: Int16Audio) -> None:
        with self._audio_lock:
            if self._ring is not None:
                self._ring.push(chunk)

    def _pop_ring(self) -> Int16Audio:
        with self._audio_lock:
            return self._ring.pop() if self._ring is not None else np.empty(0, np.int16)

    def _ring_size(self) -> int:
        with self._audio_lock:
            return self._ring.size if self._ring is not None else 0

    # ──────────────────── trigger install ─────────────────────
    def _install_triggers(self) -> None:
        def ok() -> bool:
            return not self.cfg.dual_trigger_required or (
                self._mouse_pressed and is_pressed(self.cfg.hotkey)
            )

        def toggle(src: str) -> None:
            if not ok():
                return
            on = self._recording.is_set()
            (self._recording.clear() if on else self._recording.set())
            LOGGER.info("%s (%s)", "⏸️ Paused" if on else "▶️ Recording…", src)

        add_hotkey(self.cfg.hotkey, lambda: toggle("kbd"))

        if not self.cfg.enable_mouse_trigger:
            return

        btn = {
            "left": mouse.Button.left,
            "right": mouse.Button.right,
            "middle": mouse.Button.middle,
        }[self.cfg.mouse_btn]
        self._mouse_pressed = False

        def click(_x: int, _y: int, button: mouse.Button, down: bool) -> None:
            if button is not btn:
                return
            if down:
                self._mouse_pressed = True
                if self.cfg.mouse_hold_to_record:
                    self._clear_raw_shadow()
                    self._mouse_press = time.time()
                    self._holding = False
                    self._hold_timer = threading.Timer(
                        self.cfg.mouse_hold_threshold_seconds, self._hold_start
                    )
                    self._hold_timer.start()
                else:
                    toggle("mouse")
            else:
                self._mouse_pressed = False
                if self.cfg.mouse_hold_to_record:
                    self._hold_stop()
                elif self.cfg.dual_trigger_required and self._recording.is_set():
                    toggle("mouse")

        self._mouse_listener = mouse.Listener(on_click=click)
        self._mouse_listener.start()

    def _hold_start(self) -> None:
        self._holding = True
        self._recording.set()
        LOGGER.info("▶️ Recording started (mouse hold)")

    def _hold_stop(self) -> None:
        try:
            if not self._holding:
                return
            self._recording.clear()
            self._holding = False
            LOGGER.info("⏸️ Stopped (mouse release)")
            loop = self._loop
            if loop is None:
                LOGGER.warning("Mouse release ignored before event loop is ready")
                return
            fut = asyncio.run_coroutine_threadsafe(self._flush_hold(), loop)

            def on_done(f: Future[str]) -> None:
                try:
                    text = f.result()
                except Exception as exc:
                    LOGGER.debug("Hold flush failed: %s", exc)
                    return
                if text:
                    loop.call_soon_threadsafe(self._clip_q.put_nowait, text)

            fut.add_done_callback(on_done)
        finally:
            if self._hold_timer:
                self._hold_timer.cancel()
            self._mouse_press = None

    # ──────────────────── flush helpers ───────────────────────
    async def _flush_hold(self) -> str:
        segs: list[Int16Audio] = []
        if self.cfg.use_vad:
            shadow = self._pop_raw_shadow()
            if shadow:
                segs.append(concatenate(shadow))
                if self._vad_gate:
                    self._vad_gate.force_flush()
            elif self._vad_gate:
                tail = self._vad_gate.force_flush()
                if tail is not None and tail.size:
                    segs.append(tail)
        else:
            audio = self._pop_ring()
            if not audio.size:
                return ""
            segs.append(audio)

        if not any(s.size for s in segs):
            return ""
        audio = concatenate(segs) if len(segs) > 1 else segs[0]
        txt = await self._transcribe(audio)
        return self._post.format_clipboard_text(txt) if txt else ""

    async def run_benchmark(
        self,
        seconds: float = 1.0,
        *,
        audio: Int16Audio | None = None,
        reference: str | None = None,
        runs: int = 1,
    ) -> dict[str, float | int]:
        """Measure cold-load, warm inference latency, and optional transcript quality."""
        if seconds <= 0 and audio is None:
            raise ValueError("benchmark seconds must be positive")
        if runs < 1:
            raise ValueError("benchmark runs must be positive")

        if audio is None:
            samples = max(800, int(self.cfg.sample_rate * seconds))
            t = np.arange(samples, dtype=np.float32) / self.cfg.sample_rate
            audio = (np.sin(2 * np.pi * 440 * t) * 12_000).astype(np.int16)
        else:
            samples = int(audio.size)
        audio_seconds = samples / self.cfg.sample_rate

        load_start = time.perf_counter()
        if self._backend is None:
            await self._load_model()
        model_load_ms = (time.perf_counter() - load_start) * 1_000

        latencies: list[float] = []
        text = ""
        for _ in range(runs):
            infer_start = time.perf_counter()
            text = await self._transcribe(audio)
            latencies.append((time.perf_counter() - infer_start) * 1_000)

        inference_ms = latencies[-1]
        infer_seconds = inference_ms / 1_000
        real_time_factor = (
            audio_seconds / infer_seconds if infer_seconds else float("inf")
        )

        result = {
            "audio_seconds": round(audio_seconds, 3),
            "samples": samples,
            "model_load_ms": round(model_load_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "real_time_factor": round(real_time_factor, 3),
            "chars": len(text),
        }
        if runs > 1:
            result.update(summarize_latency_ms(latencies))
        if reference is not None:
            result["wer"] = round(word_error_rate(reference, text), 4)
        LOGGER.info(
            "Benchmark %.3fs audio: load %.1f ms | infer %.1f ms | %.2fx realtime",
            result["audio_seconds"],
            result["model_load_ms"],
            result["inference_ms"],
            result["real_time_factor"],
        )
        return result

    # ──────────────────── main loop ────────────────────────────
    async def _run(self) -> None:
        clip_task = asyncio.create_task(self._clip_worker())

        if self.cfg.use_vad:
            self._vad_gate = VADGate(
                sample_rate=self.cfg.sample_rate,
                aggressiveness=int(self.cfg.vad_aggr),
                frame_duration_ms=self.cfg.vad_frame_duration_ms,
                pre_buffer_chunks=self.cfg.pre_buffer_chunks,
                post_buffer_chunks=self.cfg.post_buffer_chunks,
                consecutive_speech_frames=self.cfg.consecutive_speech_frames,
                consecutive_silence_frames=self.cfg.consecutive_silence_frames,
            )

        on_raw_chunk = self._add_to_shadow
        if not self.cfg.use_vad:
            if self._ring is None:
                raise RuntimeError("audio ring is not initialized")
            on_raw_chunk = self._add_to_ring

        async with AudioStream(
            self.cfg.sample_rate,
            chunk_ms=self.cfg.chunk_ms,
            vad_gate=self._vad_gate,
            input_device=self.cfg.input_device,
            on_raw_chunk=on_raw_chunk,
        ) as mic:
            batch: list[Int16Audio] = []
            samples = 0

            async for chunk in mic.chunks():
                if not self._recording.is_set():
                    continue

                if self._clip_q.qsize() > 8:
                    await asyncio.sleep(0.02)  # back-pressure clipboard

                if self.cfg.use_vad:
                    if chunk.size:
                        if self.cfg.mouse_hold_to_record and self._holding:
                            batch.clear()
                            samples = 0
                            continue

                        batch.append(chunk)
                        samples += chunk.size
                        self._batch_ctl.feed(chunk.size)

                        if (
                            samples >= self._batch_ctl.min_samples
                            or len(batch) >= self._batch_ctl.max_chunks
                        ):
                            audio = concatenate(batch) if len(batch) > 1 else batch[0]
                            txt = await self._transcribe(audio)
                            if txt:
                                await self._clip_q.put(
                                    self._post.format_clipboard_text(txt)
                                )
                            batch.clear()
                            samples = 0
                            if not self._holding:
                                self._clear_raw_shadow()
                else:
                    self._ensure_ring_cap(
                        int(self.cfg.max_buffer_seconds * self.cfg.sample_rate * 2)
                    )
                    if self._ring_size() >= int(
                        0.95 * self.cfg.max_buffer_seconds * self.cfg.sample_rate
                    ):
                        audio = self._pop_ring()
                        if audio.size:
                            txt = await self._transcribe(audio)
                            if txt:
                                await self._clip_q.put(
                                    self._post.format_clipboard_text(txt)
                                )

                if self._terminate.is_set():
                    break

            if batch:
                audio = concatenate(batch) if len(batch) > 1 else batch[0]
                txt = await self._transcribe(audio)
                if txt:
                    await self._clip_q.put(self._post.format_clipboard_text(txt))

        await self._clip_q.put(None)
        await clip_task

    # ──────────────────── ring growth helper ──────────────────
    def _ensure_ring_cap(self, needed: int) -> None:
        with self._audio_lock:
            if self._ring and needed > self._ring._buf.size:
                new_cap = 1 << (needed * 2 - 1).bit_length()
                LOGGER.debug("Growing ring to %d samples", new_cap)
                self._ring = _Ring(new_cap)

    # ──────────────────── public API ───────────────────────────
    async def start(self) -> None:
        await self._load_model()
        self._install_triggers()
        if not (self.cfg.enable_mouse_trigger and self.cfg.mouse_hold_to_record):
            self._recording.set()
        self._loop = asyncio.get_running_loop()
        await self._run()

    def stop(self) -> None:
        self._terminate.set()
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._hold_timer and self._hold_timer.is_alive():
            self._hold_timer.cancel()
        if self._backend is not None:
            self._model_pool.submit(self._backend.close).result()
            self._backend = None
        self._model_pool.shutdown(cancel_futures=True)
