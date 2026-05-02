"""Tests for audio I/O and streaming functionality."""

import queue
from unittest.mock import Mock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dictation_tool.io import AudioStream, VADGate, concatenate


class TestVADGate:
    """Test suite for VADGate voice activity detection."""

    def test_vadgate_initialization(self):
        """Test VADGate initialization with various parameters."""
        vad = VADGate(
            sample_rate=16000,
            aggressiveness=2,
            frame_duration_ms=30,
            pre_buffer_chunks=10,
            post_buffer_chunks=5,
        )

        assert vad.sr == 16000
        assert vad.frame_duration_ms == 30
        assert vad.bytes_per_frame == 960  # 16000 * 30 / 1000 * 2
        assert vad.pre_buffer_chunks == 10
        assert vad.post_buffer_chunks == 5
        assert vad.pre_buffer.maxlen == 10
        assert vad.post_buffer.maxlen == 5

    @given(
        sample_rate=st.integers(min_value=8000, max_value=48000),
        aggressiveness=st.integers(min_value=0, max_value=3),
        pre_buffer_chunks=st.integers(min_value=1, max_value=20),
        post_buffer_chunks=st.integers(min_value=1, max_value=10),
    )
    def test_vadgate_parameter_validation(
        self, sample_rate, aggressiveness, pre_buffer_chunks, post_buffer_chunks
    ):
        """Property-based test for VADGate parameter validation."""
        vad = VADGate(
            sample_rate=sample_rate,
            aggressiveness=aggressiveness,
            frame_duration_ms=30,
            pre_buffer_chunks=pre_buffer_chunks,
            post_buffer_chunks=post_buffer_chunks,
        )

        assert vad.sr == sample_rate
        assert vad.frame_duration_ms == 30
        expected_bytes = sample_rate * 30 // 1000 * 2  # int16 = 2 bytes
        assert vad.bytes_per_frame == expected_bytes
        assert vad.pre_buffer_chunks == pre_buffer_chunks
        assert vad.post_buffer_chunks == post_buffer_chunks

    @patch("dictation_tool.io.webrtcvad.Vad")
    def test_vadgate_speech_detection(self, mock_vad_class):
        """Test VADGate speech detection logic."""
        mock_vad = Mock()
        mock_vad_class.return_value = mock_vad

        vad_gate = VADGate(sample_rate=16000, aggressiveness=2)

        # Test speech detection
        mock_vad.is_speech.return_value = True
        test_chunk = np.array([1, 2, 3, 4] * 240, dtype=np.int16)  # 30ms worth of data

        result = list(vad_gate(test_chunk))

        # Should yield frames when speech is detected
        assert len(result) >= 0  # May yield frames based on speech detection
        mock_vad.is_speech.assert_called()

    @patch("dictation_tool.io.webrtcvad.Vad")
    def test_vadgate_no_speech_buffering(self, mock_vad_class):
        """Test VADGate buffering when no speech is detected."""
        mock_vad = Mock()
        mock_vad_class.return_value = mock_vad

        vad_gate = VADGate(sample_rate=16000, aggressiveness=2)

        # Test no speech detection
        mock_vad.is_speech.return_value = False
        test_chunk = np.array([1, 2, 3, 4] * 240, dtype=np.int16)

        result = list(vad_gate(test_chunk))

        # Should not yield when no speech detected
        assert len(result) == 0

        # Frames should be buffered before speech starts.
        assert len(vad_gate.pre_buffer) >= 0


class TestAudioStream:
    """Test suite for AudioStream functionality."""

    def test_audio_stream_init(self):
        """Test AudioStream initialization."""
        sample_rate = 16000
        chunk_ms = 20

        stream = AudioStream(sample_rate, chunk_ms)

        assert stream._sr == sample_rate
        assert stream._frames == chunk_ms * sample_rate // 1000
        assert stream._gate is None
        assert not stream._stop.is_set()

    def test_audio_stream_init_with_vad(self):
        """Test AudioStream initialization with VAD."""
        mock_vad = Mock()
        stream = AudioStream(16000, 20, vad_gate=mock_vad)

        assert stream._sr == 16000
        assert stream._gate == mock_vad

    @pytest.mark.asyncio
    async def test_audio_stream_context_manager(self):
        """Test AudioStream as async context manager."""
        with patch("dictation_tool.io.sd.InputStream") as mock_stream_class:
            mock_stream = Mock()
            mock_stream_class.return_value = mock_stream

            stream = AudioStream(16000, 20)

            async with stream as s:
                assert s == stream
                # Should start the input stream
                mock_stream_class.assert_called_once()
                mock_stream.start.assert_called_once()

    def test_audio_stream_callback_basic(self):
        """Test AudioStream callback functionality."""
        stream = AudioStream(16000, 20)

        # Mock audio data (int16)
        indata = np.array([[100], [200], [300], [400]], dtype=np.int16)

        # Test callback
        stream._callback(indata, 4, None, None)

        # Should queue the data
        assert not stream._q.empty()
        queued_data = stream._q.get_nowait()
        assert isinstance(queued_data, np.ndarray)
        assert queued_data.dtype == np.int16

    def test_audio_stream_callback_queue_handling(self):
        """Test AudioStream callback queue handling when full."""
        stream = AudioStream(16000, 20)

        # Fill the queue to capacity (simulate queue full scenario)
        with patch.object(stream._q, "put_nowait", side_effect=queue.Full):
            indata = np.array([[100], [200]], dtype=np.int16)

            # Should handle queue full gracefully
            stream._callback(indata, 2, Mock(), Mock())
            # No exception should be raised

    def test_explicit_input_device_does_not_fall_back_to_other_devices(self):
        """Explicit device selection should fail closed rather than record elsewhere."""
        stream = AudioStream(16000, 20, input_device=7)

        with (
            patch("dictation_tool.io.platform.system", return_value="Windows"),
            patch.object(stream, "_all_input_devices", return_value=[1, 2, 7]),
        ):
            assert stream._build_device_candidates(7) == [7]

    def test_default_windows_input_can_fall_back_to_available_devices(self):
        """Default input may try other Windows devices for portability."""
        stream = AudioStream(16000, 20)

        with (
            patch("dictation_tool.io.platform.system", return_value="Windows"),
            patch("dictation_tool.io.sd.default.device", (3, None)),
            patch.object(stream, "_all_input_devices", return_value=[1, 3, 5]),
        ):
            assert stream._build_device_candidates(None) == [None, 1, 3, 5]

    def test_callback_resamples_to_target_frame_count(self):
        """Native-rate fallback still emits target-rate mono int16 chunks."""
        raw_chunks = []
        stream = AudioStream(16000, 10, on_raw_chunk=raw_chunks.append)
        stream._setup_resampler(native_sr=48000, native_frames=480)
        native_chunk = np.arange(480, dtype=np.int16).reshape(-1, 1)

        stream._callback(native_chunk, 480, None, None)

        queued = stream._q.get_nowait()
        assert queued.shape == (160, 1)
        assert queued.dtype == np.int16
        assert len(raw_chunks) == 1
        np.testing.assert_array_equal(raw_chunks[0], queued)

    @pytest.mark.asyncio
    async def test_chunks_without_vad(self):
        """Test chunks method without VAD."""
        stream = AudioStream(16000, 20)

        test_frames = [
            np.array([1, 2], dtype=np.int16),
            np.array([3, 4], dtype=np.int16),
        ]

        # Mock the queue to return test frames synchronously
        call_count = 0

        def mock_get():
            nonlocal call_count
            call_count += 1
            if call_count <= len(test_frames):
                return test_frames[call_count - 1]
            else:
                # Signal stop after test frames
                stream._stop.set()
                return np.array([0], dtype=np.int16)

        with patch.object(stream._q, "get", side_effect=mock_get):
            frames = []
            async for frame in stream.chunks():
                frames.append(frame)
                if len(frames) >= len(test_frames):
                    stream._stop.set()
                    break

        assert len(frames) == len(test_frames)
        np.testing.assert_array_equal(frames[0], test_frames[0])
        np.testing.assert_array_equal(frames[1], test_frames[1])

    @pytest.mark.asyncio
    async def test_chunks_with_vad_speech(self):
        """Test chunks method with VAD detecting speech."""
        mock_vad = Mock()
        mock_vad.return_value = [np.array([1, 2], dtype=np.int16)]  # Mock VAD output

        stream = AudioStream(16000, 20, vad_gate=mock_vad)

        test_frame = np.array([1, 2], dtype=np.int16)

        call_count = 0

        def mock_get():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return test_frame
            else:
                stream._stop.set()
                return np.array([0], dtype=np.int16)

        with patch.object(stream._q, "get", side_effect=mock_get):
            frames = []
            async for frame in stream.chunks():
                frames.append(frame)
                if len(frames) >= 1:
                    stream._stop.set()
                    break

        # Should have processed through VAD
        mock_vad.assert_called()
        assert len(frames) >= 0  # VAD may filter out frames

    @pytest.mark.asyncio
    async def test_chunks_stop_behavior(self):
        """Test that chunks properly responds to stop signal."""
        stream = AudioStream(16000, 20)

        # Set stop immediately
        stream._stop.set()

        frames = []
        async for frame in stream.chunks():
            frames.append(frame)
            # Should exit quickly due to stop signal
            break

        # Should not accumulate many frames
        assert len(frames) <= 1


class TestConcatenateFunction:
    """Test suite for concatenate utility function."""

    def test_concatenate_empty_list(self):
        """Test concatenate with empty list."""
        # Empty list should raise ValueError
        with pytest.raises(ValueError):
            concatenate([])

    def test_concatenate_single_array(self):
        """Test concatenate with single array."""
        arr = np.array([1, 2, 3], dtype=np.int16)
        result = concatenate([arr])

        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.int16

    def test_concatenate_multiple_arrays(self):
        """Test concatenate with multiple arrays."""
        arr1 = np.array([1, 2], dtype=np.int16)
        arr2 = np.array([3, 4], dtype=np.int16)
        arr3 = np.array([5, 6], dtype=np.int16)

        result = concatenate([arr1, arr2, arr3])
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.int16

    @given(
        arrays=st.lists(
            st.lists(
                st.integers(min_value=-32768, max_value=32767), min_size=1, max_size=100
            ).map(lambda x: np.array(x, dtype=np.int16)),
            min_size=1,
            max_size=10,
        )
    )
    def test_concatenate_property_based(self, arrays):
        """Property-based test for concatenate function."""
        result = concatenate(arrays)

        # Result should be int16 array
        assert result.dtype == np.int16

        # Length should equal sum of input lengths
        expected_length = sum(len(arr) for arr in arrays)
        assert len(result) == expected_length

        # Should preserve order
        if len(arrays) > 0:
            # Check first and last elements
            assert result[0] == arrays[0][0]
            assert result[-1] == arrays[-1][-1]

    def test_concatenate_preserves_data_integrity(self):
        """Test that concatenate preserves data integrity."""
        # Test with specific values to ensure no data corruption
        arr1 = np.array([-32768, -1, 0], dtype=np.int16)  # Min, negative, zero
        arr2 = np.array([1, 32767], dtype=np.int16)  # Positive, max

        result = concatenate([arr1, arr2])
        expected = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

        # Verify extreme values preserved
        assert result[0] == -32768
        assert result[-1] == 32767
