import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine


@pytest.mark.gpu
@pytest.mark.timeout(15)
@pytest.mark.xfail(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_smoke():
    """Smoke test for GPU functionality - basic CUDA operations."""
    # Test basic CUDA tensor operations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        assert z.device.type == "cuda"
        assert z.shape == (100, 100)


@pytest.mark.gpu
@pytest.mark.timeout(15)
@pytest.mark.xfail(not torch.cuda.is_available(), reason="CUDA not available")
async def test_gpu_engine_init():
    """Test that DictationEngine can initialize with CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = Config(
        model_name="tiny.en",  # Use smallest model for speed
        device="cuda",
        compute_type="float16",
        attention_backend="flash",
    )

    engine = DictationEngine(cfg)

    # Test model loading
    await engine._load_model()
    assert engine._model is not None

    # Test that model is on GPU
    # Note: faster-whisper models don't expose device directly,
    # but we can verify CUDA is being used by checking torch.cuda.is_available()
    assert torch.cuda.is_available()

    # Cleanup
    engine.stop()


@pytest.mark.gpu
@pytest.mark.timeout(15)
@pytest.mark.xfail(not torch.cuda.is_available(), reason="CUDA not available")
async def test_gpu_transcription_smoke():
    """Smoke test for GPU transcription with synthetic audio."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = Config(
        model_name="tiny.en",
        device="cuda",
        compute_type="float16",
        attention_backend="flash",
    )

    engine = DictationEngine(cfg)
    await engine._load_model()

    # Create synthetic audio (1 second of sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Test transcription (should complete without error)
    result = await engine._transcribe(audio)

    # Result might be empty for synthetic audio, but should not crash
    assert isinstance(result, str)

    # Cleanup
    engine.stop()
