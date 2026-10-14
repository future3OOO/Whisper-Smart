import ctranslate2
import numpy as np
import pytest

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine


@pytest.mark.gpu
@pytest.mark.timeout(15)
@pytest.mark.skipif(
    ctranslate2.get_cuda_device_count() == 0, reason="CUDA not available"
)
def test_gpu_smoke():
    """The actual inference backend supports the configured GPU precision."""
    assert "float16" in ctranslate2.get_supported_compute_types("cuda")


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.skipif(
    ctranslate2.get_cuda_device_count() == 0, reason="CUDA not available"
)
async def test_gpu_engine_init():
    """Test that DictationEngine can initialize with CUDA device."""
    cfg = Config(
        model_name="tiny.en",  # Use smallest model for speed
        device="cuda",
        compute_type="float16",
        attention_backend="flash",
    )

    engine = DictationEngine(cfg)

    try:
        await engine._load_model()
        assert engine._backend is not None
    finally:
        engine.stop()


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.skipif(
    ctranslate2.get_cuda_device_count() == 0, reason="CUDA not available"
)
async def test_gpu_transcription_smoke():
    """Smoke test for GPU transcription with synthetic audio."""
    cfg = Config(
        model_name="tiny.en",
        device="cuda",
        compute_type="float16",
        attention_backend="flash",
    )

    engine = DictationEngine(cfg)
    # Create synthetic audio (1 second of sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    try:
        await engine._load_model()
        result = await engine._transcribe(audio)
        assert isinstance(result, str)
    finally:
        engine.stop()
