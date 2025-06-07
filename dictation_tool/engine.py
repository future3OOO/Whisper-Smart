from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Final, Callable, Deque

import numpy as np
import pyperclip
import torch
from faster_whisper import WhisperModel
from keyboard import add_hotkey, is_pressed
from pynput import mouse

try:
    import pyautogui
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
except ImportError:
    pyautogui = None

if sys.platform == "win32":
    try:
        import win32con
        import win32clipboard
    except ImportError:
        win32con = win32clipboard = None
else:
    win32con = win32clipboard = None

from .config import Config
from .io import AudioStream, VADGate, concatenate
from .utils import LOGGER, timed, retry


def _fast_win_clip(text: str) -> None:
    """Ultra-thin Win32 clipboard writer; falls back to pyperclip."""
    if not win32clipboard:
        pyperclip.copy(text)
        return
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


class _RingBuffer:
    """Lock-free, power-of-two ring-buffer for int16 audio."""
    __slots__ = ("_buf", "_mask", "_wp", "_size", "_lock")

    def __init__(self, capacity: int) -> None:
        pow2 = 1
        while pow2 < capacity: pow2 <<= 1
        self._buf = np.zeros(pow2, dtype=np.int16)
        self._mask = pow2 - 1
        self._wp = self._size = 0
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        if chunk.ndim > 1: chunk = chunk.squeeze()
        n = int(chunk.size)
        if n == 0: return
        with self._lock:
            if n >= self._buf.size:
                chunk = chunk[-self._buf.size:]; n = chunk.size
            head = min(n, self._buf.size - self._wp)
            self._buf[self._wp : self._wp + head] = chunk[:head]
            if tail := n - head: self._buf[:tail] = chunk[head:]
            self._wp = (self._wp + n) & self._mask
            self._size = min(self._size + n, self._buf.size)

    def drain(self) -> np.ndarray:
        with self._lock:
            if self._size == 0: return np.empty(0, np.int16)
            start = (self._wp - self._size) & self._mask
            if start + self._size <= self._buf.size:
                out = self._buf[start : start + self._size].copy()
            else:
                head = self._buf.size - start
                out = np.concatenate((self._buf[start:], self._buf[: self._size - head]))
            self._size = 0
            return out
    
    @property
    def size(self) -> int:
        return self._size


class ContextManager:
    """Keeps a rolling transcript history for prompt injection."""
    def __init__(self, max_history: int = 5):
        self._hist: Deque[str] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def prompt(self) -> str:
        with self._lock:
            return " ".join(self._hist)[-240:] + ". " if self._hist else ""

    def update(self, text: str) -> None:
        if text and len(text.split()) > 2:
            with self._lock:
                self._hist.append(text.strip())


class DictationEngine:
    """Mic → (optional VAD) → Whisper → clipboard."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model: WhisperModel | None = None

        max_samples = int(cfg.sample_rate * cfg.max_buffer_seconds * 1.2)
        self._f32_ws = np.empty(max_samples, dtype=np.float32)
        self._ring = _RingBuffer(max_samples) if not cfg.use_vad else None
        
        self._ctx = ContextManager()
        self._recording = asyncio.Event()
        self._terminate = asyncio.Event()
        self._mouse_listener: mouse.Listener | None = None
        self._hold_timer: threading.Timer | None = None
        self._pending_text = ""
        self._hold_recording = False
        self._mouse_press_time: float | None = None
        # CRITICAL FIX: Use a simple list, not a bounded deque, for the shadow buffer.
        self._raw_shadow: list[np.ndarray] = []
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._vad_gate: VADGate | None = None
        self._clip_pool = ThreadPoolExecutor(max_workers=1)

    async def _load_model(self) -> None:
        LOGGER.info("Loading %s …", self.cfg.model_name)
        compute = self.cfg.compute_type
        if compute == "auto":
            if self.cfg.device == "cuda" and torch.cuda.is_available():
                try:
                    compute = "int8_float16" if torch.cuda.get_device_properties(0).total_memory > 8e9 else "float16"
                except Exception:
                    compute = "float16"
            else:
                compute = "int8"
        object.__setattr__(self.cfg, "compute_type", compute)
        LOGGER.info("Auto-selected compute_type=%s", compute)

        def _sync_load() -> WhisperModel:
            threads = max(1, (os.cpu_count() or 4) // 2)
            return WhisperModel(
                self.cfg.model_name, device=self.cfg.device, compute_type=compute,
                cpu_threads=threads, num_workers=threads
            )

        with timed("model-load"):
            self._model = await asyncio.get_running_loop().run_in_executor(None, _sync_load)
        self._optimise_cuda()
        LOGGER.info("Model ready ✔")

    def _optimise_cuda(self) -> None:
        if not (self.cfg.device == "cuda" and self._model and torch.cuda.is_available()): return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            cap = torch.cuda.get_device_capability()
            ver = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
            if (self.cfg.attention_backend != "none" and hasattr(torch.backends.cuda, "enable_flash_sdp") and
                cap >= (8, 0) and ver >= (2, 0)):
                try:
                    import flash_attn; torch.backends.cuda.enable_flash_sdp(True)
                    LOGGER.info("Enabled Flash-Attention backend")
                except ImportError: pass
            if self.cfg.torch_compile_mode != "off" and ver >= (2, 1):
                torch.set_float32_matmul_precision("high")
                self._model.model = torch.compile(self._model.model, mode=self.cfg.torch_compile_mode)
                LOGGER.info("torch.compile(%s) successful", self.cfg.torch_compile_mode)
        except Exception as exc:
            LOGGER.debug("CUDA optimisation skipped: %s", exc, exc_info=True)

    def _install_triggers(self) -> None:
        self._mouse_pressed = False
        def dual_active() -> bool:
            return not self.cfg.dual_trigger_required or (is_pressed(self.cfg.hotkey) and self._mouse_pressed)
        def toggle_recording(src: str) -> None:
            if dual_active():
                if self._recording.is_set(): self._recording.clear(); LOGGER.info("⏸️  Paused (%s)", src)
                else: self._recording.set(); LOGGER.info("▶️  Recording… (%s)", src)
            elif self.cfg.dual_trigger_required: LOGGER.debug("dual-trigger: %s pressed", src)
        add_hotkey(self.cfg.hotkey, lambda: toggle_recording("keyboard"))
        if not self.cfg.enable_mouse_trigger: LOGGER.info("Trigger: %s only", self.cfg.hotkey); return
        btn_map = {"left": mouse.Button.left, "right": mouse.Button.right, "middle": mouse.Button.middle}
        mbtn = btn_map[self.cfg.mouse_btn]
        def on_click(_x: int, _y: int, button: mouse.Button, pressed: bool) -> None:
            if button is not mbtn: return
            if pressed:
                self._mouse_pressed = True
                if self.cfg.mouse_hold_to_record:
                    self._mouse_press_time = time.time(); self._hold_recording = False
                    self._pending_text = ""; self._hold_timer = threading.Timer(self.cfg.mouse_hold_threshold_seconds, self._on_hold_timeout)
                    self._raw_shadow.clear() # Clear shadow buffer at the start of a new recording
                    self._hold_timer.start()
                else: toggle_recording("mouse")
            else:
                self._mouse_pressed = False
                if self.cfg.mouse_hold_to_record: self._finish_hold_recording()
                elif self.cfg.dual_trigger_required and self._recording.is_set(): toggle_recording("mouse-release")
        self._mouse_listener = mouse.Listener(on_click=on_click, suppress=False); self._mouse_listener.start()
        msg = f"Mouse: hold {self.cfg.mouse_btn} for {self.cfg.mouse_hold_threshold_seconds:.1f}s to record" if self.cfg.mouse_hold_to_record else \
              (f"Triggers: {self.cfg.hotkey} AND {self.cfg.mouse_btn} (dual mode)" if self.cfg.dual_trigger_required else f"Triggers: {self.cfg.hotkey} or {self.cfg.mouse_btn}")
        LOGGER.info("%s%s", msg, " & auto-paste" if self.cfg.auto_paste_on_release else "")

    def _on_hold_timeout(self) -> None:
        if self._mouse_press_time and not self._hold_recording:
            self._hold_recording = True; self._recording.set()
            LOGGER.info("▶️  Recording started (held %.1fs)", self.cfg.mouse_hold_threshold_seconds)

    def _finish_hold_recording(self) -> None:
        try:
            if self._hold_recording:
                self._recording.clear(); self._hold_recording = False
                LOGGER.info("⏸️  Recording stopped (mouse released)")
                if self._main_loop:
                    fut = asyncio.run_coroutine_threadsafe(self._flush_hold_audio(), self._main_loop)
                    def _done(f: asyncio.Future) -> None:
                        try: txt: str = f.result(timeout=max(4.0, self.cfg.max_buffer_seconds * 1.5))
                        except Exception as e: LOGGER.error("Flush failed: %s", e, exc_info=True); return
                        if txt: self._clip_pool.submit(self._do_clipboard_work, txt)
                    fut.add_done_callback(_done)
        finally:
            if self._hold_timer: self._hold_timer.cancel()
            self._pending_text, self._mouse_press_time = "", None

    def _do_clipboard_work(self, txt: str) -> None:
        _fast_win_clip(txt)
        if self.cfg.auto_paste_on_release and pyautogui:
            retry(lambda: pyautogui.hotkey("ctrl", "v", interval=0)); LOGGER.debug("📋 Auto-pasted")
        else: LOGGER.debug("📋 Copied to clipboard")

    async def _transcribe(self, audio_i16: np.ndarray) -> str:
        if audio_i16.ndim > 1: audio_i16 = audio_i16.squeeze()
        n = audio_i16.size
        if n < 800: return ""
        if n > self._f32_ws.size: self._f32_ws = np.empty(int(n * 1.25), dtype=np.float32)
        
        np.multiply(audio_i16, 1 / 32768.0, out=self._f32_ws[:n], dtype=np.float32)
        audio_f32 = self._f32_ws[:n]
        if self.cfg.mic_gain != 1.0: audio_f32 *= self.cfg.mic_gain
        np.clip(audio_f32, -1.0, 1.0, out=audio_f32)
        
        with timed("infer"):
            segs, info = await asyncio.to_thread(
                self._model.transcribe, audio_f32, language=self.cfg.language,
                initial_prompt=self.cfg.initial_prompt or self._ctx.prompt(),
                beam_size=self.cfg.beam_size,
                best_of=max(self.cfg.beam_size, 5) if self.cfg.beam_size > 1 else 1,
                vad_filter=True, word_timestamps=False,
            )
        text = "".join(s.text for s in segs).strip()
        if text and getattr(info, 'avg_logprob', -1.0) < -0.8: self._ctx.update(text)
        return text

    _PUNCT_MAP: Final = {
        "dot": ".", "period": ".", "comma": ",", "colon": ":", "semicolon": ";", "dash": "-",
        "hyphen": "-", "question mark": "?", "exclamation mark": "!", "exclamation point": "!", "at sign": "@", 
        "dollar sign": "$", "percent sign": "%", "hashtag": "#",
        "open parenthesis": "(", "close parenthesis": ")", "open bracket": "[", "close bracket": "]",
        "open brace": "{", "close brace": "}", "slash": "/"
    }

    def _post_process_text(self, txt: str) -> str:
        """Applies punctuation replacements and other formatting rules."""
        if not txt: return ""
        words = txt.split()
        for i, w in enumerate(words):
            if (rep := self._PUNCT_MAP.get(w.rstrip(".,?!").lower())):
                words[i] = rep.capitalize() if w[0].isupper() else rep
        processed = " ".join(words)

        # Fix URL and path spacing (e.g., "www . google . com" -> "www.google.com")
        processed = re.sub(r'\s*([.。\\/])\s*', r'\1', processed)
        
        return processed

    def _dispatch_text(self, txt: str, hold_flush: bool = False, is_intermediate: bool = False) -> None:
        if not txt: return
        clean = self._post_process_text(txt)

        log_icon = "💬" if is_intermediate else "📝"
        LOGGER.info("%s %s", log_icon, clean)
        
        if self._hold_recording or hold_flush:
            if not is_intermediate:
                self._pending_text += (" " if self._pending_text else "") + clean
        else: self._clip_pool.submit(self._do_clipboard_work, clean)

    async def _run(self) -> None:
        if self.cfg.use_vad: self._vad_gate = VADGate(self.cfg.sample_rate, self.cfg.vad_aggr, self.cfg.vad_frame_duration_ms)
        on_chunk = self._raw_shadow.append if self.cfg.use_vad else (self._ring.write if self._ring else None)
        async with AudioStream(
            self.cfg.sample_rate, self.cfg.chunk_ms, self._vad_gate, self.cfg.input_device, on_raw_chunk=on_chunk
        ) as mic:
            async for chunk in mic.chunks():
                if self._recording.is_set():
                    if self.cfg.use_vad:
                        if chunk.size:
                            text = await self._transcribe(chunk)
                            self._dispatch_text(text, is_intermediate=True)
                    elif self._ring and self._ring.size >= self.cfg.max_buffer_seconds * self.cfg.sample_rate: await self._flush_ring()
                if self._terminate.is_set(): break
            
            tail = (self._vad_gate.force_flush() if self.cfg.use_vad and self._vad_gate else
                    self._ring.drain() if not self.cfg.use_vad and self._ring else None)
            if tail is not None and tail.size: self._dispatch_text(await self._transcribe(tail))

    async def _flush_ring(self) -> None:
        if self._ring and self._ring.size > 0: self._dispatch_text(await self._transcribe(self._ring.drain()))

    async def _flush_hold_audio(self) -> str:
        segs = [concatenate(self._raw_shadow)] if self.cfg.use_vad else [self._ring.drain() if self._ring else np.empty(0, np.int16)]
        if self.cfg.use_vad and self._vad_gate and (tail := self._vad_gate.force_flush()):
            # Only append the tail if it's not already part of the shadow buffer
            if len(segs) > 0 and tail.size > 0 and not np.array_equal(segs[0][-tail.size:], tail):
                 segs.append(tail)
        
        self._raw_shadow.clear()
        
        audio = concatenate([s for s in segs if s.size > 0])
        if not audio.size: return ""
        
        txt = await self._transcribe(audio)
        if txt: self._dispatch_text(txt, hold_flush=True)
        return self._pending_text

    async def start(self) -> None:
        await self._load_model()
        self._install_triggers()
        if not (self.cfg.enable_mouse_trigger and self.cfg.mouse_hold_to_record):
            self._recording.set(); LOGGER.info("🎙️  Recording active")
        else: LOGGER.info("🎙️  Engine ready – hold mouse to record")
        self._main_loop = asyncio.get_running_loop(); await self._run()

    async def run_benchmark(self):
        """Runs a simple latency benchmark."""
        if not self._model:
            LOGGER.error("Model not loaded, cannot run benchmark.")
            return

        LOGGER.info("Running latency benchmark...")
        dummy_audio = np.zeros(self.cfg.sample_rate * 2, dtype=np.int16)
        
        start_time = time.perf_counter()
        await self._transcribe(dummy_audio)
        end_to_end_latency = (time.perf_counter() - start_time) * 1000
        
        LOGGER.info(f"BENCHMARK RESULT: End-to-end latency: {end_to_end_latency:.2f} ms")
        if end_to_end_latency > 800:
            LOGGER.warning(f"Latency ({end_to_end_latency:.2f} ms) exceeds 800 ms target.")
        else:
            LOGGER.info("Latency is within the 800 ms target. ✓")

    def stop(self) -> None:
        self._terminate.set()
        if self._mouse_listener: self._mouse_listener.stop()
        if self._hold_timer and self._hold_timer.is_alive(): self._hold_timer.cancel()
        self._clip_pool.shutdown(wait=False, cancel_futures=True)