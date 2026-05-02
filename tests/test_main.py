"""Tests for main module."""

from pathlib import Path
from unittest import mock

import pytest

from dictation_tool import __main__


class TestMainModule:
    """Test suite for main module functionality."""

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_main_function_argument_parsing(self, mock_engine_class, mock_run):
        """Test main function argument parsing."""
        test_args = [
            "dictation_tool",
            "--device",
            "cpu",
            "--compute",
            "int8_float16",
            "--language",
            "en",
            "--initial-prompt",
            "Please transcribe:",
            "--chunk-ms",
            "25",
            "--profile",
            "profile.jsonl",
            "--hotkey",
            "alt+space",
            "--mouse-btn",
            "left",
            "--no-vad",
            "--attention-backend",
            "none",
        ]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            # Should create engine with correct config
            mock_engine_class.assert_called_once()
            config_used = mock_engine_class.call_args[0][0]
            assert config_used.device == "cpu"
            assert config_used.compute_type == "int8_float16"
            assert config_used.language == "en"
            assert config_used.initial_prompt == "Please transcribe:"
            assert config_used.chunk_ms == 25
            assert config_used.profile == Path("profile.jsonl")
            assert config_used.hotkey == "alt+space"
            assert config_used.mouse_btn == "left"
            assert config_used.use_vad is False
            assert config_used.attention_backend == "none"

            # Should call asyncio.run with engine.start()
            mock_run.assert_called_once()

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_main_function_defaults(self, mock_engine_class, mock_run):
        """Test main function uses correct defaults."""
        with mock.patch("sys.argv", ["dictation_tool"]):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            # Should create engine with default config
            mock_engine_class.assert_called_once()
            config_used = mock_engine_class.call_args[0][0]
            assert config_used.device == "cuda"
            assert config_used.compute_type == "float16"
            assert config_used.hotkey == "ctrl+space"
            assert config_used.mouse_btn == "middle"
            assert config_used.use_vad is True
            assert config_used.attention_backend == "flash"

    @mock.patch("dictation_tool.__main__.LOGGER")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_main_function_keyboard_interrupt(self, mock_engine_class, mock_logger):
        """Test main function handles KeyboardInterrupt gracefully."""
        with mock.patch("sys.argv", ["dictation_tool"]):
            with mock.patch(
                "dictation_tool.__main__.asyncio.run", side_effect=KeyboardInterrupt()
            ):
                mock_engine = mock.MagicMock()
                mock_engine_class.return_value = mock_engine

                __main__.main()

                # Should call engine.stop() on KeyboardInterrupt
                mock_engine.stop.assert_called_once()

                # Should log interruption
                mock_logger.info.assert_called_with("Interrupted - shutting down ...")

    @pytest.mark.parametrize(
        "device,expected",
        [
            ("cuda", "cuda"),
            ("cpu", "cpu"),
        ],
    )
    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_device_argument_validation(
        self, mock_engine_class, mock_run, device, expected
    ):
        """Test device argument validation."""
        test_args = ["dictation_tool", "--device", device]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            args_used = mock_engine_class.call_args[0][0]
            assert args_used.device == expected

    @pytest.mark.parametrize(
        "compute,expected",
        [
            ("float16", "float16"),
            ("int8_float16", "int8_float16"),
        ],
    )
    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_compute_argument_validation(
        self, mock_engine_class, mock_run, compute, expected
    ):
        """Test compute type argument validation."""
        test_args = ["dictation_tool", "--compute", compute]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            args_used = mock_engine_class.call_args[0][0]
            assert args_used.compute_type == expected

    @pytest.mark.parametrize(
        "mouse_btn,expected",
        [
            ("left", "left"),
            ("right", "right"),
            ("middle", "middle"),
        ],
    )
    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_mouse_button_argument_validation(
        self, mock_engine_class, mock_run, mouse_btn, expected
    ):
        """Test mouse button argument validation."""
        test_args = ["dictation_tool", "--mouse-btn", mouse_btn]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            args_used = mock_engine_class.call_args[0][0]
            assert args_used.mouse_btn == expected

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_vad_flags(self, mock_engine_class, mock_run):
        """Test VAD and attention backend flags."""
        # Test --no-vad flag
        with mock.patch("sys.argv", ["dictation_tool", "--no-vad"]):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()
            args_used = mock_engine_class.call_args[0][0]
            assert args_used.use_vad is False

        # Test --attention-backend flag
        with mock.patch("sys.argv", ["dictation_tool", "--attention-backend", "none"]):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()
            args_used = mock_engine_class.call_args[0][0]
            assert args_used.attention_backend == "none"

    def test_argument_parser_help(self):
        """Test that argument parser can be called for help."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["dictation_tool", "--help"]):
                __main__.main()

    def test_invalid_device_choice(self):
        """Test invalid device choice raises SystemExit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["dictation_tool", "--device", "invalid"]):
                __main__.main()

    def test_invalid_compute_choice(self):
        """Test invalid compute choice raises SystemExit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["dictation_tool", "--compute", "invalid"]):
                __main__.main()

    def test_invalid_mouse_button_choice(self):
        """Test invalid mouse button choice raises SystemExit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["dictation_tool", "--mouse-btn", "invalid"]):
                __main__.main()

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_mouse_hold_to_record_arguments(self, mock_engine_class, mock_run):
        """Test mouse hold-to-record argument parsing."""
        test_args = [
            "dictation_tool",
            "--toggle-mouse-record",
            "--mouse-hold-threshold",
            "1.5",
            "--auto-paste",
        ]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            args_used = mock_engine_class.call_args[0][0]
            assert args_used.mouse_hold_to_record is False
            assert args_used.mouse_hold_threshold_seconds == 1.5
            assert args_used.auto_paste_on_release is True

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_bench_runs_benchmark_instead_of_live_engine(
        self, mock_engine_class, mock_run
    ):
        """Benchmark mode should run the measurement path and not start dictation."""
        mock_engine = mock.MagicMock()
        mock_engine_class.return_value = mock_engine

        with mock.patch("sys.argv", ["dictation_tool", "--device", "cpu", "--bench"]):
            __main__.main()

        mock_engine.run_benchmark.assert_called_once()
        mock_engine.start.assert_not_called()
        mock_run.assert_called_once()

    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_bench_models_runs_each_requested_model(self, mock_engine_class, mock_run):
        """Benchmark matrix mode should instantiate one engine per model."""
        engines = [mock.MagicMock(), mock.MagicMock()]
        mock_engine_class.side_effect = engines

        with mock.patch(
            "sys.argv",
            [
                "dictation_tool",
                "--device",
                "cpu",
                "--bench",
                "--bench-models",
                "large-v3-turbo,distil-large-v3",
            ],
        ):
            __main__.main()

        models = [call.args[0].model_name for call in mock_engine_class.call_args_list]
        assert models == ["large-v3-turbo", "distil-large-v3"]
        assert engines[0].run_benchmark.called
        assert engines[1].run_benchmark.called
        assert engines[0].start.call_count == 0
        assert engines[1].start.call_count == 0
        assert mock_run.call_count == 2

    @mock.patch("dictation_tool.__main__.load_wav_mono_int16")
    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_bench_passes_audio_reference_runs_and_model_vad(
        self, mock_engine_class, mock_run, mock_load_audio
    ):
        """Benchmark mode should pass real-audio options to run_benchmark."""
        mock_engine = mock.MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_audio = mock.MagicMock()
        mock_load_audio.return_value = mock_audio

        with mock.patch(
            "sys.argv",
            [
                "dictation_tool",
                "--device",
                "cpu",
                "--bench",
                "--bench-audio",
                "fixture.wav",
                "--bench-reference",
                "hello world",
                "--bench-runs",
                "3",
                "--no-model-vad",
            ],
        ):
            __main__.main()

        cfg = mock_engine_class.call_args.args[0]
        assert cfg.model_vad_filter is False
        mock_load_audio.assert_called_once()
        mock_engine.run_benchmark.assert_called_once_with(
            audio=mock_audio,
            reference="hello world",
            runs=3,
        )
