"""
dictation_tool.engine  –  v3.3-titan (2025-06-08)

Key features
────────────
• Unlimited shadow buffer during mouse-hold (guarantees full 60 s capture)
• Soft RMS gate (8 000) – keeps very quiet consonants
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
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Deque, Tuple, Optional, Literal, Any

import numpy as np
import pyperclip
import torch
from faster_whisper import WhisperModel
from keyboard import add_hotkey, is_pressed, send as kb_send
from pynput import mouse

from .config import Config
from .io import AudioStream, VADGate, concatenate
from .utils import LOGGER, timed
from .utils.profile import prof

# ══════════════════════════════════ Clipboard helpers ═════════════════════════
if sys.platform == "win32":
    try:
        import win32clipboard as _wc  # type: ignore
        import win32con as _wcon      # type: ignore
    except ImportError:               # pywin32 not installed
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
                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
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

    __slots__ = ("_buf", "_mask", "_head", "_tail", "_full", "_view")

    def __init__(self, cap: int) -> None:
        cap = 1 << (cap - 1).bit_length()
        self._buf = np.zeros(cap, np.int16)
        self._mask = cap - 1
        self._head = self._tail = 0
        self._full = False
        self._view = memoryview(self._buf).cast("h")

    def push(self, chunk: np.ndarray) -> None:
        if chunk.ndim > 1:
            chunk = chunk.ravel()
        n = int(chunk.size)
        if not n:
            return
        cap = self._view.shape[0]
        if n >= cap:                       # keep only the last <cap> samples
            self._view[:] = chunk[-cap:]
            self._head = self._tail = 0
            self._full = True
            return
        head, end = self._head, self._head + n
        if end <= cap:
            self._view[head:end] = chunk
        else:
            cut = cap - head
            self._view[head:] = chunk[:cut]
            self._view[: end - cap] = chunk[cut:]
        self._head = end & self._mask
        if self._full or (self._head <= self._tail < head):
            self._tail = self._head
            self._full = True

    def pop(self) -> np.ndarray:
        if self._head == self._tail and not self._full:
            return np.empty(0, np.int16)
        cap = self._view.shape[0]
        if self._head > self._tail or self._full:
            out = self._buf[self._tail : self._head].copy()
        else:
            out = np.concatenate((self._buf[self._tail :], self._buf[: self._head]))
        self._tail = self._head
        self._full = False
        return out

    @property
    def size(self) -> int:
        return self._view.shape[0] if self._full else (self._head - self._tail) & self._mask

# ══════════════════════════════════ Punctuation map ═══════════════════════════
class _Punct:
    _MAP = {
        ("at", "sign"): "@",
        ("dot",): ".",
        ("comma",): ",",
        ("colon",): ":",
        ("semicolon",): ";",
        ("question", "mark"): "?",
        ("exclamation", "point"): "!",
        ("exclamation", "mark"): "!",
        ("dash",): "-",
        ("hyphen",): "-",
        ("slash",): "/",
        ("open", "parenthesis"): "(",
        ("close", "parenthesis"): ")",
        ("open", "bracket"): "[",
        ("close", "bracket"): "]",
        ("open", "brace"): "{",
        ("close", "brace"): "}",
        ("dollar", "sign"): "$",
        ("percent", "sign"): "%",
        ("hash", "tag"): "#",
        ("pound", "sign"): "#",
    }

    def __init__(self) -> None:
        self._patterns = [
            (re.compile(r"\b" + r"\s+".join(map(re.escape, k)) + r"\b", re.I), v)
            for k, v in self._MAP.items()
        ]
        self._patterns.sort(key=lambda kv: -kv[0].pattern.count(r"\s"))
        # keep line-feeds intact; collapse only spaces/tabs
        self._spc_before = re.compile(r"[ \t]+([,.:;?!()[\]{}])")
        self._spc_after = re.compile(r"([(\[{])[ \t]+")

    def __call__(self, text: str) -> str:
        out = text
        for pat, rep in self._patterns:
            out = pat.sub(rep, out)
        out = self._spc_before.sub(r"\1", out)
        out = self._spc_after.sub(r"\1", out)
        return out.strip()

# ═════════════════════════ Thread-safe context ════════════════════════════════
class _Context:
    """Rolling prompt history (mutex-guarded)."""

    def __init__(self, hist: int = 5) -> None:
        self._buf: Deque[str] = deque(maxlen=hist)
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
            LOGGER.info("Adaptive batch tuned → %d samples | %d chunks",
                        self.min_samples, self.max_chunks)
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
            LOGGER.warning("Clipboard fast-path failed: %s – fallback", exc)
            await loop.run_in_executor(self._pool, pyperclip.copy, text)
        self._last = (text, time.time())

# ───────────────────────── command cleanup ──────────────────────────
_CMD_SUBS: tuple[tuple[re.Pattern, str], ...] = (
    # single line break  – eat optional punctuation / spaces after the cue
    (re.compile(r"\b(?:new\s+line|line\s*break|newline)\b[ \t]*[.,!?;:]?[ \t]*",
                re.I), "\n"),
    # blank line (paragraph) – same idea, but keep the double LF
    (re.compile(r"\bnew\s+paragraph\b[ \t]*[.,!?;:]?[ \t]*", re.I), "\n\n"),
    (re.compile(r"\bbullet\s+point\b", re.I), "\n• "),
    # strip runs of smart quotes, plain quote, back-tick, or � (U+FFFD)
    (re.compile(r'[\u201C\u201D"`\uFFFD]+'), ""),
)

_SPACES_AROUND_DOT_AT = re.compile(r"[ \t\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]*([@.])[ \t\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]*")

_SIGNOFFS = ("kind regards", "best regards", "regards", "cheers")
SIGNOFF_PAT = re.compile(
    r"(^|\n)(%s)\b[,.\s]*" % "|".join(_SIGNOFFS),
    re.I,
)

# ══════════════════════════ Dictation Engine ══════════════════════════════════
class DictationEngine:
    """Microphone → (VAD) → Whisper → clipboard."""

    _EMAIL_RE = re.compile(r"\b([\w.-]+)\s+at(?:\s+sign)?\s+([\w.-]+)\s+dot\s+com\b",
                          re.I)
    _URL_DOT = re.compile(r"\b([a-zA-Z0-9_-]+)\s+\.\s+([a-zA-Z0-9_-]+)")

    # ───────────────────────── init ──────────────────────────
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cap = int(cfg.sample_rate * cfg.max_buffer_seconds * 2)
        self._f32 = np.empty(cap, np.float32)
        self._i32 = np.empty(cap, np.int32)
        self._ring = _Ring(cap) if not cfg.use_vad else None

        self._model: WhisperModel | None = None
        self._recording = asyncio.Event()
        self._terminate = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._clip = _Clipboard()
        self._clip_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=10)
        self._punct = _Punct()
        self._ctx = _Context()
        self._batch_ctl = _BatchCtl(cfg)

        # triggers
        self._mouse_listener: Optional[mouse.Listener] = None
        self._mouse_press: Optional[float] = None
        self._hold_timer: Optional[threading.Timer] = None
        self._holding = False

        # shadow buffer
        self._raw_shadow: Deque[np.ndarray] = (
            deque()
            if cfg.mouse_hold_to_record
            else deque(maxlen=int(cfg.max_buffer_seconds * 1000 / cfg.chunk_ms))
        )
        self._vad_gate: Optional[VADGate] = None

        self._rms: dict[Tuple[int, int, int, int], int] = {}

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

        def _sync() -> WhisperModel:
            th = max(1, (os.cpu_count() or 4) // 2)
            return WhisperModel(
                self.cfg.model_name,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=th,
                num_workers=th,
            )

        LOGGER.info("Loading %s …", self.cfg.model_name)
        with prof("model_load", self.cfg.profile), timed("model-load"):
            self._model = await asyncio.get_running_loop().run_in_executor(None, _sync)

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
    async def _transcribe(self, audio: np.ndarray) -> str:
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

        with prof("infer", self.cfg.profile), timed("infer"):
            segs, info = self._model.transcribe(
                self._f32[:n],
                language=self.cfg.language,
                initial_prompt=self.cfg.initial_prompt or self._ctx.prompt(),
                beam_size=self.cfg.beam_size,
                best_of=self.cfg.best_of,
                temperature=self.cfg.temperature,
                vad_filter=True,
                word_timestamps=False,
            )

        txt = "".join(s.text for s in segs).strip()
        if txt and getattr(info, "avg_logprob", -1.0) > -0.6:
            self._ctx.push(txt)

        # deterministic clean-ups -------------------------------------------------
        txt = self._EMAIL_RE.sub(r"\1@\2.com", self._URL_DOT.sub(r"\1.\2", txt))
        for pat, rep in _CMD_SUBS:
            txt = pat.sub(rep, txt)
        
        # ── NEW: collapse duplicate commas (word "comma" + real comma) ─────────
        txt = re.sub(r',\s*,+', ',', txt)
        
        # ── NEW: if a comma sneaks in *before* the paragraph break, make it a '.' ─
        txt = re.sub(r',\s*\n\n', '.\n\n', txt)
        
        # ── NEW: add full stop before paragraph break when *no* punctuation spoken ─
        txt = re.sub(r'([^\s.,!?;:])\s*\n\n', r'\1.\n\n', txt)
        
        # final space trim around @ and .
        txt = _SPACES_AROUND_DOT_AT.sub(r'\1', txt)
        
        # safety-pass: remove spaces or tabs (NOT new-lines) that may survive
        txt = re.sub(r'@[ \t]+', '@', txt)   # john @ gmail → john@gmail
        txt = re.sub(r'\.[ \t]+', '.', txt)  # gmail . com  → gmail.com
        
        # -------------------------------------------------------------------------
        # 8. sentence-/signature-polish  (run *after* all previous tweaks)
        # -------------------------------------------------------------------------
        
        # 8-a  normalise common e-mail sign-offs
        txt = SIGNOFF_PAT.sub(lambda m: f"{m.group(1)}{m.group(2).title()},\n", txt)
        
        # 8-b  capitalise first alphabetical char of every logical line
        txt = re.sub(
            r"(^|\n)([• \t]*)([a-z])",
            lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
            txt,
        )
        
        return txt

    # ──────────────────── shadow helper ──────────────────────
    def _add_to_shadow(self, chunk: np.ndarray) -> None:
        self._raw_shadow.append(chunk)

    # ──────────────────── trigger install ─────────────────────
    def _install_triggers(self) -> None:
        def ok() -> bool:
            return (
                not self.cfg.dual_trigger_required
                or (self._mouse_pressed and is_pressed(self.cfg.hotkey))
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
                    self._raw_shadow.clear()     # start fresh
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
            fut = asyncio.run_coroutine_threadsafe(self._flush_hold(), self._loop)
            fut.add_done_callback(
                lambda f: self._loop.call_soon_threadsafe(
                    self._clip_q.put_nowait, f.result() or ""
                )
            )
        finally:
            if self._hold_timer:
                self._hold_timer.cancel()
            self._mouse_press = None

    # ──────────────────── flush helpers ───────────────────────
    async def _flush_hold(self) -> str:
        segs: list[np.ndarray] = []
        if self.cfg.use_vad:
            if self._raw_shadow:
                segs.append(concatenate(self._raw_shadow))
                self._raw_shadow.clear()
            if self._vad_gate:
                tail = self._vad_gate.force_flush()
                if tail is not None and tail.size:
                    segs.append(tail)
        else:
            segs.append(self._ring.pop())

        if not any(s.size for s in segs):
            return ""
        audio = concatenate(segs) if len(segs) > 1 else segs[0]
        txt = await self._transcribe(audio)
        return self._punct(txt) if txt else ""

    # ──────────────────── main loop ────────────────────────────
    async def _run(self) -> None:
        clip_task = asyncio.create_task(self._clip_worker())

        if self.cfg.use_vad:
            self._vad_gate = VADGate(
                sample_rate=self.cfg.sample_rate,
                aggressiveness=self.cfg.vad_aggr,
                frame_duration_ms=self.cfg.vad_frame_duration_ms,
                pre_buffer_chunks=self.cfg.pre_buffer_chunks,
                post_buffer_chunks=self.cfg.post_buffer_chunks,
                consecutive_speech_frames=self.cfg.consecutive_speech_frames,
                consecutive_silence_frames=self.cfg.consecutive_silence_frames,
            )

        async with AudioStream(
            self.cfg.sample_rate,
            chunk_ms=self.cfg.chunk_ms,
            vad_gate=self._vad_gate,
            input_device=self.cfg.input_device,
            on_raw_chunk=self._add_to_shadow if self.cfg.use_vad else self._ring.push,
        ) as mic:
            batch: list[np.ndarray] = []
            samples = 0

            async for chunk in mic.chunks():
                if not self._recording.is_set():
                    continue

                if self._clip_q.qsize() > 8:
                    await asyncio.sleep(0.02)   # back-pressure clipboard

                if self.cfg.use_vad:
                    if chunk.size:
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
                                await self._clip_q.put(self._punct(txt))
                            batch.clear()
                            samples = 0
                            if not self._holding:
                                self._raw_shadow.clear()
                else:
                    self._ensure_ring_cap(
                        self.cfg.max_buffer_seconds * self.cfg.sample_rate * 2
                    )
                    if self._ring.size >= int(
                        0.95 * self.cfg.max_buffer_seconds * self.cfg.sample_rate
                    ):
                        audio = self._ring.pop()
                        if audio.size:
                            txt = await self._transcribe(audio)
                            if txt:
                                await self._clip_q.put(self._punct(txt))

                if self._terminate.is_set():
                    break

            if batch:
                audio = concatenate(batch) if len(batch) > 1 else batch[0]
                txt = await self._transcribe(audio)
                if txt:
                    await self._clip_q.put(self._punct(txt))

        await self._clip_q.put(None)
        await clip_task

    # ──────────────────── ring growth helper ──────────────────
    def _ensure_ring_cap(self, needed: int) -> None:
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
