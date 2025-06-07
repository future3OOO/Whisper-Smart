from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, Union

ComputeType = Literal[
    "auto",          # resolve at runtime for best perf/accuracy balance
    "float16",
    "int8_float16",
    "float32",
    "int8_float32",
    "int16",
    "int8",
    "mem-eff",
]
TorchCompileMode = Literal["off", "default", "reduce-overhead", "max-autotune"]
AttentionBackend = Literal["none", "flash", "mem-eff"]


@dataclass(slots=True)
class Config:
    """Central runtime configuration."""

    # Whisper
    model_name: str = "large-v3"
    compute_type: ComputeType = "auto"
    device: str = "cuda"                           # "cuda" | "cpu"
    beam_size: int = 5
    language: str | None = None
    initial_prompt: str | None = None

    # Audio / runtime
    sample_rate: int = 16_000
    chunk_ms: int = 10                             # Lower latency default
    # VAD aggressiveness: 0-3 or "auto"
    vad_aggr: Union[int, Literal["auto"]] = "auto"
    use_vad: bool = True
    max_buffer_seconds: float = 10.0
    input_device: str | None = None                # Audio input device name/index
    mic_gain: float = 1.0                     # stay ≤5; higher values may clip

    # VADGate - Pre-buffer system
    vad_frame_duration_ms: int = 30               # WebRTC VAD frame duration
    pre_buffer_chunks: int = 10                   # Chunks to keep before speech
    post_buffer_chunks: int = 5                   # Chunks to keep after speech
    consecutive_speech_frames: int = 3            # Frames needed to trigger
    consecutive_silence_frames: int = 8           # Frames needed to end
    
    # Triggers
    hotkey: str = "ctrl+alt+space"
    mouse_btn: Literal["left", "right", "middle"] = "right"
    enable_mouse_trigger: bool = False
    dual_trigger_required: bool = False
    
    # Mouse hold-to-record behavior
    mouse_hold_to_record: bool = True
    mouse_hold_threshold_seconds: float = 0.2
    # clipboard options
    copy_on_release: bool = False
    auto_paste_on_release: bool = False

    # Retry Logic with Adaptive Temperature (GPU-optimized defaults)
    enable_retry_logic: bool = True
    retry_temperatures: Sequence[float] | None = None
    max_retries: int = 3
    min_confidence_threshold: float = 0.6  # stricter default

    # GPU Performance Optimization
    torch_compile_mode: TorchCompileMode = "off" # Safer default
    attention_backend: AttentionBackend = "flash"

    # Misc
    log_dir: Path = Path.home() / ".dictation_tool" / "logs"

    def __post_init__(self):
        """Initialize immutable defaults and validate configuration."""

        # ❶ auto-pick VAD aggressiveness
        if self.vad_aggr == "auto":
            # A practical rule-of-thumb: level 2 is less prone to chop syllables
            object.__setattr__(self, "vad_aggr", 2)

        # Set immutable retry temperatures
        if self.retry_temperatures is None:
            # Two-pass on GPU → quicker; CPU keeps three passes.
            if self.device == "cuda":
                self.retry_temperatures = (0.0, 0.6)  # 2-pass is enough; final 0.9 added in engine
            else:
                self.retry_temperatures = (0.0, 0.4, 0.7)
        
        # Validate optimisation flags against the runtime
        self._validate_torch_compatibility()
    
    def _validate_torch_compatibility(self) -> None:
        """Downgrade optimisation flags when the runtime cannot honour them."""
        try:
            import torch

            # Strip build metadata like +cu121 before splitting version
            version_clean = torch.__version__.split("+")[0]
            major, minor, *_ = map(int, version_clean.split("."))
            
            if self.torch_compile_mode != "off" and (major, minor) < (2, 1):
                print(
                    "torch.compile needs PyTorch ≥ 2.1 – disabling.",
                    file=sys.stderr,
                )
                object.__setattr__(self, "torch_compile_mode", "off")
                
            if self.attention_backend != "none" and (major, minor) < (2, 0):
                print(
                    "Flash/Mem-Eff attention need PyTorch ≥ 2.0 – disabling.",
                    file=sys.stderr,
                )
                object.__setattr__(self, "attention_backend", "none")
                
        except ImportError:
            pass  # handled elsewhere