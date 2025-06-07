"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest

from dictation_tool.config import Config


class TestConfig:
    """Test suite for Config dataclass."""

    def test_config_defaults(self):
        """Test Config default values."""
        cfg = Config()

        # Whisper settings
        assert cfg.model_name == "large-v3"
        assert cfg.compute_type == "float16"
        assert cfg.device == "cuda"
        assert cfg.attention_backend == "flash"

        # Audio/runtime settings
        assert cfg.sample_rate == 16_000
        assert cfg.chunk_ms == 30
        assert cfg.vad_aggr == 3
        assert cfg.use_vad is True
        assert cfg.max_buffer_seconds == 12.0

        # VADGate settings
        assert cfg.vad_frame_duration_ms == 30
        assert cfg.pre_buffer_chunks == 10
        assert cfg.post_buffer_chunks == 5
        assert cfg.consecutive_speech_frames == 3
        assert cfg.consecutive_silence_frames == 8

        # Trigger settings
        assert cfg.hotkey == "ctrl+space"
        assert cfg.mouse_btn == "middle"
        assert cfg.enable_mouse_trigger is True
        assert cfg.dual_trigger_required is False

        # Retry logic settings
        assert cfg.enable_retry_logic is True
        assert cfg.retry_temperatures == (0.0, 0.3, 0.6)  # GPU-optimized default
        assert cfg.max_retries == 3
        assert cfg.min_confidence_threshold == 0.30

        # GPU optimization settings
        assert cfg.torch_compile_mode == "default"
        assert cfg.attention_backend == "flash"

        # Transcription settings
        assert cfg.language == "en"
        assert cfg.initial_prompt is None

        # Misc settings
        assert cfg.log_dir == Path.home() / ".dictation_tool" / "logs"

    def test_config_custom_values(self):
        """Test Config with custom values."""
        custom_log_dir = Path(tempfile.gettempdir()) / "test_logs"

        cfg = Config(
            model_name="base",
            compute_type="int8_float16",
            device="cpu",
            attention_backend="none",
            sample_rate=8000,
            chunk_ms=120,
            vad_aggr=3,
            use_vad=False,
            max_buffer_seconds=5.0,
            pre_buffer_chunks=5,
            post_buffer_chunks=3,
            consecutive_speech_frames=2,
            consecutive_silence_frames=6,
            hotkey="alt+space",
            mouse_btn="right",
            enable_mouse_trigger=False,
            dual_trigger_required=True,
            enable_retry_logic=False,
            retry_temperatures=(0.0, 0.5, 1.0),
            max_retries=2,
            min_confidence_threshold=0.8,
            torch_compile_mode="reduce-overhead",
            language="en",
            initial_prompt="Please transcribe:",
            log_dir=custom_log_dir,
            input_device=None,
            mic_gain=1.0,
        )

        # Verify custom values
        assert cfg.model_name == "base"
        assert cfg.compute_type == "int8_float16"
        assert cfg.device == "cpu"
        assert cfg.attention_backend == "none"
        assert cfg.sample_rate == 8000
        assert cfg.chunk_ms == 120
        assert cfg.vad_aggr == 3
        assert cfg.use_vad is False
        assert cfg.max_buffer_seconds == 5.0
        assert cfg.pre_buffer_chunks == 5
        assert cfg.post_buffer_chunks == 3
        assert cfg.consecutive_speech_frames == 2
        assert cfg.consecutive_silence_frames == 6
        assert cfg.hotkey == "alt+space"
        assert cfg.mouse_btn == "right"
        assert cfg.enable_mouse_trigger is False
        assert cfg.dual_trigger_required is True
        assert cfg.enable_retry_logic is False
        assert cfg.retry_temperatures == (0.0, 0.5, 1.0)  # Should be tuple after __post_init__
        assert cfg.max_retries == 2
        assert cfg.min_confidence_threshold == 0.8
        assert cfg.torch_compile_mode == "reduce-overhead"
        assert cfg.attention_backend == "none"
        assert cfg.language == "en"
        assert cfg.initial_prompt == "Please transcribe:"
        assert cfg.log_dir == custom_log_dir

    @pytest.mark.parametrize("compute_type", ["float16", "int8_float16"])
    def test_config_compute_type_validation(self, compute_type):
        """Test that valid compute types are accepted."""
        cfg = Config(compute_type=compute_type)
        assert cfg.compute_type == compute_type

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_config_device_validation(self, device):
        """Test that valid devices are accepted."""
        cfg = Config(device=device)
        assert cfg.device == device

    @pytest.mark.parametrize("mouse_btn", ["left", "right", "middle"])
    def test_config_mouse_button_validation(self, mouse_btn):
        """Test that valid mouse buttons are accepted."""
        cfg = Config(mouse_btn=mouse_btn)
        assert cfg.mouse_btn == mouse_btn

    @pytest.mark.parametrize("vad_aggr", [0, 1, 2, 3])
    def test_config_vad_aggressiveness_validation(self, vad_aggr):
        """Test that valid VAD aggressiveness levels are accepted."""
        cfg = Config(vad_aggr=vad_aggr)
        assert cfg.vad_aggr == vad_aggr

    def test_config_immutable_with_slots(self):
        """Test that Config uses slots (more memory efficient)."""
        cfg = Config()

        # Should have __slots__ attribute
        assert hasattr(Config, "__slots__")

        # Should not be able to add arbitrary attributes
        with pytest.raises(AttributeError):
            cfg.random_attribute = "should fail"

    def test_config_path_handling(self):
        """Test that log_dir handles Path objects correctly."""
        path_obj = Path("/tmp/test_logs")
        cfg = Config(log_dir=path_obj)

        # Should accept Path object
        assert isinstance(cfg.log_dir, Path)
        assert cfg.log_dir == path_obj
