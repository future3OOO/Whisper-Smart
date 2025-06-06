from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterable
from enum import Enum
from typing import Callable

import numpy as np
import sounddevice as sd
import webrtcvad

from .utils import LOGGER, timed

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

    def __call__(self, chunk: np.ndarray) -> Iterable[np.ndarray]:
        """Process audio chunk and yield voiced segments with optimal buffering.
        
        Returns segments only when speech has definitively ended,
        ensuring complete utterances are captured.
        """
        pcm = chunk.tobytes()
        results: list[np.ndarray] = []
        
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
    
    def _process_frame(self, frame: bytes, is_speech: bool) -> np.ndarray | None:
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
    
    def _create_segment(self) -> np.ndarray:
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
        
    def force_flush(self) -> np.ndarray | None:
        """Force flush any pending speech buffer (e.g., on session end)."""
        if self.speech_buffer:
            segment = self._create_segment()
            self._reset_for_next_utterance()
            return segment
        return None

class AudioStream:
    """Mic reader → optional VAD → async chunk generator."""

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int = 10,
        vad_gate: VADGate | None = None,
        input_device: str | None = None,
        on_raw_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """
        `on_raw_chunk` receives every *raw* 10 ms frame (for shadow buffering).
        """
        self._sr = sample_rate
        self._frames = int(self._sr * chunk_ms / 1000)
        self._gate = vad_gate
        self._input_device = input_device
        self._on_raw_chunk = on_raw_chunk
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()

    async def __aenter__(self) -> AudioStream:
        device_params = {}
        if self._input_device is not None:
            device_params['device'] = self._input_device
            
        self._stream = sd.InputStream(  # type: ignore[attr-defined]
            samplerate=self._sr,
            channels=1,
            dtype="int16",
            blocksize=self._frames,
            callback=self._callback,
            **device_params
        )
        self._stream.start()
        device_info = f" (device: {self._input_device})" if self._input_device else ""
        LOGGER.info("🎙️  Mic @ %d Hz%s - press hot-key to toggle", self._sr, device_info)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._stream.stop()
        self._stream.close()

    # ── internals ────────────────────────────────────────────────────────
    def _callback(self, indata: np.ndarray, _frames: int, *_) -> None:
        try:
            if self._on_raw_chunk:
                # Keep a copy of *every* frame before it goes to VAD
                self._on_raw_chunk(indata.copy())
            self._q.put_nowait(indata.copy())
        except queue.Full:
            pass  # reader is behind, drop a frame

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            raw = await loop.run_in_executor(None, self._q.get)
            if self._gate:
                for voiced in self._gate(raw):
                    yield voiced
            else:
                yield raw


def concatenate(chunks: Iterable[np.ndarray]) -> np.ndarray:
    with timed("concat"):
        return np.concatenate(chunks, dtype=np.int16)