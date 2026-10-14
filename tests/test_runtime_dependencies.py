"""The inference runtime works without the model-conversion framework."""

import os
import subprocess
import sys


def test_runtime_starts_without_pytorch():
    script = """
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("PyTorch is not a runtime dependency", name=fullname)

sys.meta_path.insert(0, NoTorch())
from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine
from dictation_tool import __main__
engine = DictationEngine(Config(device='cpu'))
engine.stop()
assert 'torch' not in sys.modules
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "DICT__DEVICE")
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
