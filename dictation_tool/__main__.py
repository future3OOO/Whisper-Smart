"""
dictation_tool.main - Titan launch wrapper
────────────────────────────────────────────
Launches the DictationEngine with a flexible CLI that exposes every runtime
flag Titan understands, while applying safe defaults for new users.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from dictation_tool.benchmark import load_wav_mono_int16, parse_model_matrix
from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine
from dictation_tool.prompts import PRESETS
from dictation_tool.utils import LOGGER


# ───────────────────────── helper parsers ────────────────────────────────────
def _parse_floats(csv: str) -> tuple[float, ...]:
    try:
        return tuple(float(x) for x in csv.split(","))
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(f"invalid float list: {exc}") from exc


def _parse_vad_aggr(val: str) -> int | str:
    if val.lower() == "auto":
        return "auto"
    try:
        i = int(val)
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError("must be 0-3 or 'auto'") from exc
    if i not in (0, 1, 2, 3):
        raise argparse.ArgumentTypeError("must be 0-3 or 'auto'")
    return i


# ───────────────────────── argparse boilerplate ──────────────────────────────
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dictation_tool",
        description="Realtime dictation powered by Whisper (Titan edition)",
    )

    # Core
    p.add_argument("--model", default="large-v3-turbo", help="Whisper checkpoint")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument(
        "--compute",
        choices=(
            "auto",
            "float16",
            "int8_float16",
            "float32",
            "int8_float32",
            "int16",
            "int8",
        ),
        default="auto",
        help="Quantisation / dtype",
    )
    p.add_argument("--language", help="Force transcription language (e.g. 'en')")

    # Accuracy & prompting
    g_acc = p.add_argument_group("accuracy")
    g_acc.add_argument("--beam-size", type=int, default=5)
    g_acc.add_argument("--best-of", type=int, default=1)
    g_acc.add_argument("--temperature", type=float, default=0.0)
    g_acc.add_argument(
        "--retry-temperatures",
        type=_parse_floats,
        help="e.g. '0.0,0.6,0.9' (overrides config default)",
    )
    g_acc.add_argument(
        "--no-model-vad",
        dest="model_vad_filter",
        action="store_false",
        help="Disable faster-whisper internal VAD filtering",
    )

    # Prompting
    g_prompt = p.add_argument_group("prompting")
    g_prompt.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        help="Domain preset (overridden by --initial-prompt)",
    )
    g_prompt.add_argument("--terms", help="Comma-separated extra vocab")
    g_prompt.add_argument("--initial-prompt", help="Full prompt, bypass presets")

    # Audio / VAD
    g_audio = p.add_argument_group("audio / VAD")
    g_audio.add_argument("--chunk-ms", type=int, default=10)
    g_audio.add_argument("--no-vad", action="store_true", help="Disable VAD")
    g_audio.add_argument("--vad-aggr", type=_parse_vad_aggr, default="auto")
    g_audio.add_argument("--input-device")
    g_audio.add_argument("--mic-gain", type=float, default=1.0)
    g_audio.add_argument("--max-buffer-s", type=float, default=60.0)

    # Triggers
    g_trig = p.add_argument_group("triggers")
    g_trig.add_argument("--hotkey", default="ctrl+space")
    g_trig.add_argument(
        "--mouse-btn", choices=("left", "right", "middle"), default="middle"
    )
    g_trig.set_defaults(enable_mouse_trigger=False)
    g_trig.add_argument("--mouse", dest="enable_mouse_trigger", action="store_true")
    g_trig.add_argument("--no-mouse", dest="enable_mouse_trigger", action="store_false")
    g_trig.add_argument("--dual-trigger-required", action="store_true")
    g_trig.add_argument(
        "--toggle-mouse-record", dest="mouse_hold_to_record", action="store_false"
    )
    g_trig.add_argument("--mouse-hold-threshold", type=float, default=0.2)

    # Clipboard
    g_clip = p.add_mutually_exclusive_group()
    g_clip.add_argument(
        "--auto-paste",
        dest="auto_paste_on_release",
        action="store_true",
        help="Copy AND press Ctrl+V on release",
    )
    g_clip.add_argument(
        "--copy-only",
        dest="copy_on_release",
        action="store_true",
        help="Copy only, user pastes manually",
    )

    # GPU / optimisation
    g_gpu = p.add_argument_group("gpu")
    g_gpu.add_argument(
        "--torch-compile-mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="off",
    )
    g_gpu.add_argument(
        "--attention-backend", choices=("none", "flash", "mem-eff"), default="flash"
    )

    # Misc / log level
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--profile", help="Write model-load and inference timings to JSONL")
    p.add_argument(
        "--bench-models",
        help="Comma-separated model list for --bench; use 'default' for the model matrix",
    )
    p.add_argument("--bench-audio", help="16-bit PCM WAV file to benchmark")
    p.add_argument("--bench-reference", help="Reference transcript for WER scoring")
    p.add_argument("--bench-runs", type=int, default=1, help="Inference runs per model")
    p.add_argument(
        "--bench", action="store_true", help="Run latency benchmark and exit"
    )

    # hidden tiny model flag for CI
    p.add_argument("--tiny-model", action="store_true", help=argparse.SUPPRESS)
    return p


# ───────────────────────── prompt resolver ───────────────────────────────────
def _resolve_prompt(args: argparse.Namespace) -> str | None:
    """Returns the final prompt string or None."""
    initial_prompt = cast(str | None, args.initial_prompt)
    if initial_prompt:
        return initial_prompt

    preset = cast(str | None, args.preset)
    if not preset:
        return None

    template, default_terms = PRESETS[preset]
    terms = cast(str | None, args.terms)
    vocab = terms if terms else default_terms
    return template.format(terms=vocab)


def _config_with_model(cfg: Config, model_name: str) -> Config:
    if hasattr(cfg, "model_copy"):
        return cfg.model_copy(update={"model_name": model_name})
    return cfg.copy(update={"model_name": model_name})


def _run_benchmark_matrix(
    cfg: Config,
    models: tuple[str, ...],
    *,
    audio_path: str | None = None,
    reference: str | None = None,
    runs: int = 1,
) -> None:
    audio = (
        load_wav_mono_int16(Path(audio_path), cfg.sample_rate) if audio_path else None
    )
    for model_name in models:
        engine = DictationEngine(_config_with_model(cfg, model_name))
        try:
            result = asyncio.run(
                engine.run_benchmark(audio=audio, reference=reference, runs=runs)
            )
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "model_vad_filter": cfg.model_vad_filter,
                        **result,
                    },
                    sort_keys=True,
                )
            )
        finally:
            engine.stop()


# ───────────────────────── main entry ────────────────────────────────────────
def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    # logging level
    if args.verbose and args.quiet:
        parser.error("choose only one of --verbose / --quiet")
    LOGGER.setLevel("DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO")

    # CUDA guard
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                parser.error("CUDA requested but not available; use --device cpu")
        except ImportError:
            parser.error("PyTorch not installed - cannot enable CUDA")

    # model override for CI
    model_name = "tiny.en" if args.tiny_model else args.model

    # compute auto-logic
    compute_type = args.compute
    if compute_type == "auto":
        compute_type = "float16" if args.device == "cuda" else "int8"

    # safety clamps
    if args.mic_gain > 5.0:
        LOGGER.warning("Mic gain clamped to 5.0")
        args.mic_gain = 5.0
    if not 0.0 <= args.temperature <= 1.0:
        parser.error("--temperature must be 0-1")
    if args.beam_size < 1 or args.best_of < 1:
        parser.error("--beam-size and --best-of must be ≥ 1")

    # resolve prompt
    final_prompt = _resolve_prompt(args)

    # build config dataclass
    cfg = Config(
        model_name=model_name,
        device=args.device,
        compute_type=compute_type,
        language=args.language,
        initial_prompt=final_prompt,
        model_vad_filter=args.model_vad_filter,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        max_retries=3,
        min_confidence_threshold=0.6,
        vad_aggr=2 if args.vad_aggr == "auto" else args.vad_aggr,
        retry_temperatures=args.retry_temperatures,
        chunk_ms=args.chunk_ms,
        max_buffer_seconds=args.max_buffer_s,
        input_device=args.input_device,
        mic_gain=args.mic_gain,
        hotkey=args.hotkey,
        enable_mouse_trigger=getattr(args, "enable_mouse_trigger", True),
        mouse_btn=args.mouse_btn,
        dual_trigger_required=args.dual_trigger_required,
        mouse_hold_to_record=args.mouse_hold_to_record,
        mouse_hold_threshold_seconds=args.mouse_hold_threshold,
        auto_paste_on_release=args.auto_paste_on_release,
        copy_on_release=args.copy_on_release,
        use_vad=not args.no_vad,
        torch_compile_mode=args.torch_compile_mode,
        attention_backend=args.attention_backend,
        profile=args.profile,
    )

    if args.bench:
        if args.bench_runs < 1:
            parser.error("--bench-runs must be >= 1")
        models = (
            parse_model_matrix(args.bench_models)
            if args.bench_models
            else (cfg.model_name,)
        )
        try:
            _run_benchmark_matrix(
                cfg,
                models,
                audio_path=args.bench_audio,
                reference=args.bench_reference,
                runs=args.bench_runs,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            LOGGER.info("Interrupted - shutting down ...")
        return

    engine = DictationEngine(cfg)
    try:
        asyncio.run(engine.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        LOGGER.info("Interrupted - shutting down ...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
