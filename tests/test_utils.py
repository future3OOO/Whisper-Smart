"""Tests for utility functions."""

from __future__ import annotations

import logging
import time
from unittest import mock

import pytest

from dictation_tool.utils import LOGGER, clamp_tokens, retry, timed


def test_logger_configuration():
    """Test that logger is properly configured."""
    assert LOGGER.name == "dictation_tool"
    assert LOGGER.level == logging.INFO
    assert len(LOGGER.handlers) == 1

    handler = LOGGER.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    # The stream might be redirected during testing, so just check it exists
    assert hasattr(handler, "stream")


def test_timed_context_manager(caplog):
    """Test the timed context manager."""
    # Set logger to DEBUG to capture timing messages
    LOGGER.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        with timed("test-operation"):
            time.sleep(0.01)  # Sleep for 10ms

    # Should have logged the timing
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "test-operation took" in record.message
    assert "ms" in record.message
    assert record.levelname == "DEBUG"

    # Reset logger level
    LOGGER.setLevel(logging.INFO)


def test_timed_with_exception(caplog):
    """Test timed context manager when exception occurs."""
    LOGGER.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        try:
            with timed("failing-operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass

    # Should still log timing even when exception occurs
    assert len(caplog.records) == 1
    assert "failing-operation took" in caplog.records[0].message

    LOGGER.setLevel(logging.INFO)


def test_retry_success():
    """Test retry with successful function."""
    call_count = 0

    def success_fn():
        nonlocal call_count
        call_count += 1
        return "success"

    result = retry(success_fn)
    assert result == "success"
    assert call_count == 1


def test_retry_with_failures():
    """Test retry with function that fails then succeeds."""
    call_count = 0

    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Failure {call_count}")
        return "success"

    with mock.patch("time.sleep") as mock_sleep:
        result = retry(flaky_fn)

    assert result == "success"
    assert call_count == 3

    # Should have called sleep with exponential backoff + jitter
    assert mock_sleep.call_count == 2

    # Check that sleep times are in expected ranges (base ± 20% jitter)
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

    # First retry: 0.05 * 2^0 * jitter (0.8-1.2) = 0.04-0.06
    assert 0.04 <= sleep_calls[0] <= 0.06

    # Second retry: 0.05 * 2^1 * jitter (0.8-1.2) = 0.08-0.12
    assert 0.08 <= sleep_calls[1] <= 0.12


def test_retry_custom_attempts():
    """Test retry with custom number of attempts."""
    call_count = 0

    def always_fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Always fails")

    with pytest.raises(RuntimeError, match="Always fails"):
        with mock.patch("time.sleep"):
            retry(always_fail, attempts=3)

    assert call_count == 3


def test_retry_custom_backoff():
    """Test retry with custom backoff."""
    call_count = 0

    def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ValueError("Failure")
        return "success"

    with mock.patch("time.sleep") as mock_sleep:
        result = retry(fail_twice, backoff=0.1)

    assert result == "success"
    assert call_count == 3

    # Should use custom backoff with jitter
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

    # First retry: 0.1 * 2^0 * jitter (0.8-1.2) = 0.08-0.12
    assert 0.08 <= sleep_calls[0] <= 0.12

    # Second retry: 0.1 * 2^1 * jitter (0.8-1.2) = 0.16-0.24
    assert 0.16 <= sleep_calls[1] <= 0.24


def test_retry_all_attempts_fail():
    """Test retry when all attempts fail."""
    call_count = 0

    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError(f"Failed attempt {call_count}")

    with pytest.raises(ConnectionError, match="Failed attempt 5"):
        with mock.patch("time.sleep"):
            retry(always_fail)  # Default 5 attempts

    assert call_count == 5


def test_retry_logs_debug_messages(caplog):
    """Test that retry logs debug messages for failures."""
    LOGGER.setLevel(logging.DEBUG)
    call_count = 0

    def fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("First failure")
        return "success"

    with caplog.at_level(logging.DEBUG):
        with mock.patch("time.sleep"):
            result = retry(fail_once)

    assert result == "success"

    # Should have logged the retry
    debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
    retry_logs = [r for r in debug_records if "Retry" in r.message]
    assert len(retry_logs) == 1
    assert "Retry 1" in retry_logs[0].message
    assert "First failure" in retry_logs[0].message

    LOGGER.setLevel(logging.INFO)


def test_retry_with_different_exception_types():
    """Test retry behavior with different exception types."""
    call_count = 0

    def mixed_exceptions():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Value error")
        elif call_count == 2:
            raise ConnectionError("Connection error")
        return "success"

    with mock.patch("time.sleep"):
        result = retry(mixed_exceptions)

    assert result == "success"
    assert call_count == 3


def test_retry_preserves_last_exception():
    """Test that retry raises the last exception when all attempts fail."""

    def fail_with_different_messages():
        import random

        msg = f"Error {random.randint(1, 1000)}"
        raise ValueError(msg)

    # Mock random to make it deterministic
    with mock.patch("random.randint", side_effect=[1, 2, 3, 4, 5]):
        with mock.patch("time.sleep"):
            with pytest.raises(ValueError, match="Error 5"):
                retry(fail_with_different_messages)


def test_clamp_tokens():
    """Test clamp_tokens utility function."""
    # Test short message (no clamping needed)
    short_msg = "Hello world"
    assert clamp_tokens(short_msg, 100) == "Hello world"

    # Test exact limit (no clamping needed)
    exact_msg = "x" * 100
    assert clamp_tokens(exact_msg, 100) == exact_msg

    # Test message needing clamping
    long_msg = "x" * 200
    clamped = clamp_tokens(long_msg, 100)
    assert len(clamped) == 100
    assert clamped.endswith("...")
    assert clamped == "x" * 97 + "..."

    # Test default limit
    very_long_msg = "a" * 2000
    clamped_default = clamp_tokens(very_long_msg)
    assert len(clamped_default) == 1000
    assert clamped_default.endswith("...")

    # Test edge cases
    assert clamp_tokens("", 10) == ""
    assert clamp_tokens("abc", 3) == "abc"
    assert clamp_tokens("abcd", 3) == "..."
