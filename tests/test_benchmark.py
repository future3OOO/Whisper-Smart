"""Tests for benchmark and accuracy helpers."""

import wave

import numpy as np

from dictation_tool.benchmark import (
    DEFAULT_MODEL_MATRIX,
    load_wav_mono_int16,
    parse_model_matrix,
    summarize_latency_ms,
    word_error_rate,
)


def test_parse_model_matrix_uses_default_candidates():
    assert parse_model_matrix(None) == DEFAULT_MODEL_MATRIX


def test_parse_model_matrix_accepts_csv_and_removes_duplicates():
    assert parse_model_matrix("large-v3, medium.en,large-v3") == (
        "large-v3",
        "medium.en",
    )


def test_summarize_latency_ms_reports_tail_metrics():
    summary = summarize_latency_ms([10.0, 20.0, 30.0, 40.0])

    assert summary == {
        "count": 4,
        "min_ms": 10.0,
        "p50_ms": 25.0,
        "p95_ms": 40.0,
        "max_ms": 40.0,
    }


def test_word_error_rate_scores_reference_against_candidate():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello, world!", "hello world") == 0.0
    assert word_error_rate("hello brave world", "hello world") == 1 / 3


def test_load_wav_mono_int16_resamples_pcm_fixture(tmp_path):
    path = tmp_path / "fixture.wav"
    audio = np.array([0, 1000, -1000, 0], dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(audio.tobytes())

    loaded = load_wav_mono_int16(path, sample_rate=16000)

    assert loaded.dtype == np.int16
    assert loaded.size == 8
