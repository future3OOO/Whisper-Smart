"""
dictation_tool.config - unified runtime settings (v3.3-titan)
--------------------------------------------------------------
• Env-var overrides:  DICT__MODEL_NAME=base.en   (any field)
• All legacy fields kept, plus new titan options
• Type-checked, range-checked, and Torch-aware
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------------ enums
ComputeType = Literal[
    "auto",
    "float16",
    "int8_float16",
    "float32",
    "int8_float32",
    "int16",
    "int8",
]
TorchCompileMode = Literal["off", "default", "reduce-overhead", "max-autotune"]
AttentionBackend = Literal["none", "flash", "mem-eff"]
ClipboardMode = Literal["async", "thread", "direct"]


# ------------------------------------------------------------------ Config
class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DICT__", extra="forbid")

    # ============ Whisper ============ -----------------------------
    model_name: str = Field("large-v3-turbo", description="Whisper checkpoint")
    compute_type: ComputeType = "auto"
    device: str = Field(
        default_factory=lambda: (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        )
    )
    beam_size: int = 5
    best_of: int = 1
    temperature: float = 0.0
    language: str | None = None
    initial_prompt: str | None = None
    model_vad_filter: bool = True

    # ============ Audio / runtime ============ ----------------------
    sample_rate: int = 16_000
    chunk_ms: int = Field(10, ge=5, le=1000)
    use_vad: bool = True
    vad_aggr: int | Literal["auto"] = "auto"
    max_buffer_seconds: float = Field(60.0, gt=5.0, le=120.0)  # titan default 20 s
    input_device: str | None = None
    mic_gain: float = Field(1.0, gt=0.1, le=10.0)

    # --- VADGate tuning -------------------------------------------
    vad_frame_duration_ms: int = 30
    pre_buffer_chunks: int = 10
    post_buffer_chunks: int = 5
    consecutive_speech_frames: int = 3
    consecutive_silence_frames: int = 8

    # ============ Titan batching  ============ ----------------------
    batch_min_samples: int = 16_000
    batch_max_chunks: int = 6
    adaptive_batching: bool = True

    # ============ Triggers ============ ----------------------------
    hotkey: str = "ctrl+alt+space"
    mouse_btn: Literal["left", "right", "middle"] = "right"
    enable_mouse_trigger: bool = False
    dual_trigger_required: bool = False

    # --- Hold-to-record -------------------------------------------
    mouse_hold_to_record: bool = True
    mouse_hold_threshold_seconds: float = Field(0.2, gt=0.05, le=2.0)

    # ============ Clipboard ================= -----------------------
    copy_on_release: bool = False  # kept for bw-compat (internal only)
    auto_paste_on_release: bool = False
    clipboard_mode: ClipboardMode = "async"

    # ============ Retry logic =============== -----------------------
    enable_retry_logic: bool = True
    retry_temperatures: Sequence[float] | None = None
    max_retries: int = Field(3, ge=1, le=10)
    min_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)

    # ============ GPU / perf =============== -----------------------
    torch_compile_mode: TorchCompileMode = "off"
    attention_backend: AttentionBackend = "flash"

    # ============ Misc ===================== -----------------------
    log_dir: Path = Path.home() / ".dictation_tool" / "logs"
    profile: Path | None = Field(
        None, description="Path to JSONL profiler output; if None profiling is off"
    )

    # ----------------- Validators ---------------------------------
    @field_validator("vad_aggr", mode="before")
    @classmethod
    def _auto_vad(cls, v: int | str) -> int:
        if v == "auto":
            return 2
        return int(v)

    @field_validator("beam_size")
    @classmethod
    def _min_beam(cls, v: int) -> int:
        if v < 1:
            print("beam_size must be >=1 - resetting to 1.", file=sys.stderr)
            return 1
        return v

    @field_validator("best_of")
    @classmethod
    def _min_best_of(cls, v: int) -> int:
        return max(v, 1)

    @field_validator("temperature")
    @classmethod
    def _clamp_temp(cls, v: float) -> float:
        return max(0.0, min(v, 1.0))

    @field_validator("batch_max_chunks")
    @classmethod
    def _batch_chunks_range(cls, v: int) -> int:
        if not 2 <= v <= 10:
            raise ValueError("batch_max_chunks must be 2-10")
        return v

    @model_validator(mode="after")
    def _post_init_logic(self) -> Config:
        # -------- best_of vs beam_size ----------
        if self.beam_size == 1 and self.best_of > 1:
            print(
                "best_of ignored when beam_size == 1 - setting best_of = 1.",
                file=sys.stderr,
            )
            object.__setattr__(self, "best_of", 1)

        # -------- retry temp defaults -----------
        if self.retry_temperatures is None:
            retry_temperatures = (
                (0.0, 0.6) if self.device == "cuda" else (0.0, 0.4, 0.7)
            )
            object.__setattr__(self, "retry_temperatures", retry_temperatures)

        # -------- torch compatibility -----------
        try:
            import torch

            major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
            if self.torch_compile_mode != "off" and (major, minor) < (2, 1):
                print("torch.compile needs PyTorch >=2.1 - disabling.", file=sys.stderr)
                object.__setattr__(self, "torch_compile_mode", "off")

            if self.attention_backend != "none" and (major, minor) < (2, 0):
                print(
                    "Flash/Mem-Eff attention need PyTorch >=2.0 - disabling.",
                    file=sys.stderr,
                )
                object.__setattr__(self, "attention_backend", "none")
        except ImportError:
            pass

        return self
