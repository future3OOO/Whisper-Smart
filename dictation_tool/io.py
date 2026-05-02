from __future__ import annotations

import asyncio
import platform
import queue
import threading
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from enum import Enum
from types import TracebackType
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from .utils import LOGGER, timed

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import webrtcvad  # type: ignore[import-untyped]

Int16Audio = NDArray[np.int16]
Float64Array = NDArray[np.float64]

__all__ = [
    "AudioStream",
    "VADGate",
    "VADState",
    "concatenate",
]


class VADState(Enum):
    """Voice Activity Detection state machine."""

    SILENCE = "silence"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"


class VADGate:
    """Advanced Voice Activity Detector with pre-buffering and state machine."""

    # ... (no changes to this class) ...
    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        frame_duration_ms: int = 30,
        pre_buffer_chunks: int = 10,
        post_buffer_chunks: int = 5,
        consecutive_speech_frames: int = 3,
        consecutive_silence_frames: int = 50,
    ) -> None:
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.bytes_per_frame = self.sr * frame_duration_ms // 1000 * 2  # int16

        # Pre-buffer configuration
        self.pre_buffer_chunks = pre_buffer_chunks
        self.post_buffer_chunks = post_buffer_chunks
        self.consecutive_speech_frames = consecutive_speech_frames
        self.consecutive_silence_frames = consecutive_silence_frames

        # State tracking
        self.state = VADState.SILENCE
        self.speech_frame_count = 0
        self.silence_frame_count = 0

        # Buffers - using deque for O(1) operations
        self.pre_buffer: deque[bytes] = deque(maxlen=pre_buffer_chunks)
        self.speech_buffer: list[bytes] = []
        self.post_buffer: deque[bytes] = deque(maxlen=post_buffer_chunks)

        # Performance tracking
        self._frames_processed = 0

    def reset(self) -> None:
        """Reset VAD state and buffers."""
        self.state = VADState.SILENCE
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.pre_buffer.clear()
        self.speech_buffer.clear()
        self.post_buffer.clear()

    def get_statistics(self) -> dict[str, int | str]:
        """Get processing statistics for monitoring."""
        return {
            "frames_processed": self._frames_processed,
            "pre_buffer_size": len(self.pre_buffer),
            "speech_buffer_size": len(self.speech_buffer),
            "post_buffer_size": len(self.post_buffer),
            "current_state": self.state.value,
        }

    def __call__(self, chunk: Int16Audio) -> Iterable[Int16Audio]:
        """Process audio chunk and yield voiced segments with optimal buffering.

        Returns segments only when speech has definitively ended,
        ensuring complete utterances are captured.
        """
        pcm = chunk.tobytes()
        results: list[Int16Audio] = []

        # Process each frame in the chunk
        for i in range(0, len(pcm), self.bytes_per_frame):
            frame = pcm[i : i + self.bytes_per_frame]
            if len(frame) < self.bytes_per_frame:
                continue

            self._frames_processed += 1

            # VAD detection
            try:
                is_speech = self.vad.is_speech(frame, self.sr)
            except Exception:
                # Handle invalid frame gracefully
                is_speech = False

            # State machine processing
            segment = self._process_frame(frame, is_speech)
            if segment is not None and len(segment) > 0:
                results.append(segment)

        return results

    def _process_frame(self, frame: bytes, is_speech: bool) -> Int16Audio | None:
        """Process a single frame through the VAD state machine."""
        if self.state == VADState.SILENCE:
            if is_speech:
                self.speech_frame_count += 1
                self.silence_frame_count = 0

                if self.speech_frame_count >= self.consecutive_speech_frames:
                    # Speech detected! Transition to speech state
                    self.state = VADState.SPEECH_DETECTED

                    # Add pre-buffer and current frame to speech buffer
                    self.speech_buffer.extend(self.pre_buffer)
                    self.speech_buffer.append(frame)

                    # Clear pre-buffer
                    self.pre_buffer.clear()
                else:
                    # Not enough consecutive speech frames yet
                    self.pre_buffer.append(frame)
            else:
                # Continue in silence
                self.speech_frame_count = 0
                self.pre_buffer.append(frame)

        elif self.state == VADState.SPEECH_DETECTED:
            if is_speech:
                # Continue speech
                self.silence_frame_count = 0
                self.speech_buffer.append(frame)

                # Clear any post-buffer
                self.post_buffer.clear()
            else:
                # Potential end of speech
                self.silence_frame_count += 1
                self.post_buffer.append(frame)

                if self.silence_frame_count >= self.consecutive_silence_frames:
                    # Speech definitely ended - create segment immediately
                    complete_segment = self._create_segment()
                    self._reset_for_next_utterance()
                    return complete_segment

        return None

    def _create_segment(self) -> Int16Audio:
        """Create a complete speech segment from buffers."""
        if not self.speech_buffer:
            return np.array([], dtype=np.int16)

        # Combine speech buffer with post-buffer
        all_frames = self.speech_buffer + list(self.post_buffer)

        # Convert to numpy arrays and concatenate
        frame_arrays = [np.frombuffer(frame, dtype=np.int16) for frame in all_frames]
        return np.concatenate(frame_arrays, dtype=np.int16)

    def _reset_for_next_utterance(self) -> None:
        """Reset state for detecting the next speech utterance."""
        self.state = VADState.SILENCE
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.speech_buffer.clear()
        self.post_buffer.clear()

    def force_flush(self) -> Int16Audio | None:
        """Force flush any pending speech buffer (e.g., on session end)."""
        if self.speech_buffer:
            segment = self._create_segment()
            self._reset_for_next_utterance()
            return segment
        return None


class AudioStream:
    """Mic reader -> optional VAD -> async chunk generator.

    Captures at the device's native sample rate and resamples to
    ``sample_rate`` in the audio callback when the two differ.
    Resampling uses numpy linear interpolation (~8 us per 10 ms chunk).
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int = 10,
        vad_gate: VADGate | None = None,
        input_device: str | int | None = None,
        on_raw_chunk: Callable[[Int16Audio], None] | None = None,
    ) -> None:
        """
        `on_raw_chunk` receives every sample-rate-normalized frame for shadow buffering.
        """
        self._sr = sample_rate
        self._chunk_ms = chunk_ms
        self._frames = int(self._sr * chunk_ms / 1000)
        self._gate = vad_gate
        self._input_device = input_device
        self._on_raw_chunk = on_raw_chunk
        self._q: queue.Queue[Int16Audio] = queue.Queue(maxsize=64)
        self._stop = threading.Event()

        # Set during _open_stream when native rate != target rate
        self._native_sr: int = sample_rate
        self._resample_idx: Float64Array | None = None
        self._native_arange: Float64Array | None = None

    async def __aenter__(self) -> AudioStream:
        self._stream = self._open_stream(self._input_device)
        self._stream.start()
        return self

    def _open_stream(self, device: str | int | None) -> sd.InputStream:
        """Open an InputStream, falling back to native-rate + resample.

        Strategy:
          1. Try the requested device (or system default) at target rate.
          2. On Windows, also try every WASAPI input at target rate.
          3. If all fail, open the best candidate at its native rate
             and resample each chunk via np.interp (~8 us / 10 ms).
        """
        candidates = self._build_device_candidates(device)

        # --- Pass 1: try target sample rate directly ---
        last_err: Exception | None = None
        for dev in candidates:
            try:
                stream = self._try_open(dev, self._sr)
                self._native_sr = self._sr
                self._resample_idx = None
                self._log_mic(dev, self._sr, resample=False)
                return stream
            except sd.PortAudioError as exc:
                last_err = exc
                LOGGER.debug("Device %s @ %d Hz failed: %s", dev, self._sr, exc)

        # --- Pass 2: open at native rate, resample in callback ---
        for dev in candidates:
            native_sr = self._device_native_sr(dev)
            if native_sr is None or native_sr == self._sr:
                continue
            try:
                native_frames = int(native_sr * self._chunk_ms / 1000)
                stream = self._try_open(dev, native_sr, blocksize=native_frames)
                self._setup_resampler(native_sr, native_frames)
                self._log_mic(dev, native_sr, resample=True)
                return stream
            except sd.PortAudioError as exc:
                last_err = exc
                LOGGER.debug(
                    "Device %s @ %d Hz (native) failed: %s", dev, native_sr, exc
                )

        raise sd.PortAudioError(
            f"No usable input device found (last error: {last_err})"
        )

    def _try_open(
        self,
        device: str | int | None,
        sr: int,
        blocksize: int | None = None,
    ) -> sd.InputStream:
        params: dict[str, str | int] = {}
        if device is not None:
            params["device"] = device
        return sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=blocksize or self._frames,
            callback=self._callback,
            **params,
        )

    def _build_device_candidates(
        self, device: str | int | None
    ) -> list[int | str | None]:
        if device is not None:
            return [device]

        candidates: list[int | str | None] = []
        candidates.append(None)

        if platform.system() == "Windows":
            try:
                input_devices = self._all_input_devices()
            except sd.PortAudioError as exc:
                LOGGER.debug("Unable to enumerate input devices: %s", exc)
                input_devices = []
            for idx in input_devices:
                if idx not in candidates:
                    candidates.append(idx)
        return candidates

    def _setup_resampler(self, native_sr: int, native_frames: int) -> None:
        target_frames = int(self._sr * self._chunk_ms / 1000)
        self._native_sr = native_sr
        self._resample_idx = np.linspace(
            0, native_frames - 1, target_frames, dtype=np.float64
        )
        self._native_arange = np.arange(native_frames, dtype=np.float64)

    def _log_mic(self, dev: str | int | None, sr: int, *, resample: bool) -> None:
        label = dev if dev is not None else "default"
        if resample:
            LOGGER.info(
                "Mic @ %d Hz (device: %s) -> resample to %d Hz - press hot-key to toggle",
                sr,
                label,
                self._sr,
            )
        else:
            LOGGER.info(
                "Mic @ %d Hz (device: %s) - press hot-key to toggle",
                sr,
                label,
            )

    @staticmethod
    def _device_native_sr(device: str | int | None) -> int | None:
        try:
            if device is not None:
                info = sd.query_devices(device, "input")
            else:
                info = sd.query_devices(kind="input")
            return int(info["default_samplerate"])
        except Exception:
            return None

    @staticmethod
    def _all_input_devices() -> list[int]:
        """Return all input device indices, WASAPI first, then others."""
        hostapis = sd.query_hostapis()
        api_priority = {"WASAPI": 0, "DirectSound": 1, "MME": 2}
        devices = cast(Iterable[dict[str, Any]], sd.query_devices())
        inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]

        def sort_key(pair: tuple[int, dict[str, Any]]) -> tuple[int, Any]:
            _, d = pair
            api_name = hostapis[d["hostapi"]]["name"]
            pri = next((v for k, v in api_priority.items() if k in api_name), 10)
            return (pri, d.get("index", 0))

        inputs.sort(key=sort_key)
        return [i for i, _ in inputs]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._stream.stop()
        self._stream.close()

    # ── internals ────────────────────────────────────────────────────────
    def _callback(self, indata: Int16Audio, _frames: int, *_args: object) -> None:
        try:
            chunk = indata.copy()
            if self._resample_idx is not None and self._native_arange is not None:
                chunk = (
                    np.interp(
                        self._resample_idx,
                        self._native_arange,
                        chunk[:, 0].astype(np.float64),
                    )
                    .astype(np.int16)
                    .reshape(-1, 1)
                )
            if self._on_raw_chunk:
                self._on_raw_chunk(chunk.copy())
            self._q.put_nowait(chunk)
        except queue.Full:
            pass

    async def chunks(self) -> AsyncIterator[Int16Audio]:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            raw = await loop.run_in_executor(None, self._q.get)
            if self._gate:
                for voiced in self._gate(raw):
                    yield voiced
            else:
                yield raw


def concatenate(chunks: Iterable[Int16Audio]) -> Int16Audio:
    with timed("concat"):
        return np.concatenate(list(chunks), dtype=np.int16)
