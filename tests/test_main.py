"""Tests for main module."""

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
            "--prompt",
            "Please transcribe:",
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
            with mock.patch("dictation_tool.__main__.asyncio.run", side_effect=KeyboardInterrupt()):
                mock_engine = mock.MagicMock()
                mock_engine_class.return_value = mock_engine

                __main__.main()

                # Should call engine.stop() on KeyboardInterrupt
                mock_engine.stop.assert_called_once()

                # Should log interruption
                mock_logger.info.assert_called_with("Interrupted by user")

    @pytest.mark.parametrize(
        "device,expected",
        [
            ("cuda", "cuda"),
            ("cpu", "cpu"),
        ],
    )
    @mock.patch("dictation_tool.__main__.asyncio.run")
    @mock.patch("dictation_tool.__main__.DictationEngine")
    def test_device_argument_validation(self, mock_engine_class, mock_run, device, expected):
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
    def test_compute_argument_validation(self, mock_engine_class, mock_run, compute, expected):
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
            "--mouse-hold-to-record",
            "--mouse-hold-threshold",
            "1.5",
            "--auto-paste",
        ]

        with mock.patch("sys.argv", test_args):
            mock_engine = mock.MagicMock()
            mock_engine_class.return_value = mock_engine

            __main__.main()

            args_used = mock_engine_class.call_args[0][0]
            assert args_used.mouse_hold_to_record is True
            assert args_used.mouse_hold_threshold_seconds == 1.5
            assert args_used.auto_paste_on_release is True
