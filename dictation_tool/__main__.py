from __future__ import annotations

import argparse
import asyncio
import sys
# from typing import Sequence # No longer needed

from dictation_tool.config import Config
from dictation_tool.engine import DictationEngine
from dictation_tool.prompts import PRESETS
from dictation_tool.utils import LOGGER, timed


def _parse_temperatures(value: str) -> tuple[float, ...]:
    """Parse comma-separated temperatures into a tuple of floats."""
    try:
        return tuple(map(float, value.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid temperature value: {e}")


def main():
    """Main entry point for the dictation tool."""
    parser = argparse.ArgumentParser(description="Realtime dictation (Whisper large-v3)")
    
    # Core settings
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute", default="auto", choices=("auto", "float16", "int8_float16", "float32", "int8_float32", "int16", "int8"))
    parser.add_argument("--model", default="large-v3", help="Name of the Whisper model to use")
    parser.add_argument("--language", default=None, help="Force transcription language (e.g., 'en', 'es')")
    
    # Accuracy Tuning (from research)
    prompt_group = parser.add_argument_group("Accuracy and Prompting")
    prompt_group.add_argument("--preset", choices=PRESETS.keys(), help="Use a pre-defined prompt for a specific domain (e.g., 'general', 'programming').")
    prompt_group.add_argument("--initial-prompt", type=str, default=None, help="Provide a full, static prompt to override all presets and context.")
    prompt_group.add_argument("--beam-size", type=int, default=5, help="Beam size for transcription (1 is fastest, 5 is more accurate).")
    
    # Audio settings
    parser.add_argument("--chunk-ms", type=int, default=10, help="Audio chunk size in milliseconds")
    parser.add_argument("--no-vad", action="store_true", help="Disable Voice Activity Detection")
    
    def _vad_aggr(value: str) -> int | str:
        """Return int 0-3 or the literal string 'auto'."""
        if value.lower() == "auto":
            return "auto"
        try:
            v = int(value)
            if v not in (0, 1, 2, 3):
                raise ValueError
            return v
        except ValueError:
            raise argparse.ArgumentTypeError("must be 0-3 or 'auto'")

    parser.add_argument("--vad-aggr", type=_vad_aggr, default="auto",
                        help="VAD aggressiveness (0-3) or 'auto'")
                        
    parser.add_argument("--retry-on-degraded", action="store_true", help="Enable automatic retries on low-confidence transcriptions.")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence threshold for transcription quality.")

    # Hidden flag for benchmarking
    parser.add_argument("--bench", action="store_true", help="Run a latency benchmark after transcription.")

    parser.add_argument(
        "--retry-temperatures",
        type=_parse_temperatures,
        default=None,
        help="Comma-separated temperatures for retry logic (e.g., '0.0,0.6,0.9')",
    )
    parser.add_argument("--input-device", help="Audio input device name or index")
    parser.add_argument("--mic-gain", type=float, default=1.0, help="Microphone gain. Keep < 5. Aim for RMS ~-20dBFS in verbose mode.")
    parser.add_argument("--max-buffer-s", type=float, default=10.0, help="Maximum audio buffer size in seconds")
    
    # Trigger settings
    parser.add_argument("--hotkey",
                        default="ctrl+space",
                        help="Global hotkey to toggle recording")

    # ── Mouse trigger settings ──────────────────────────────────────────
    parser.add_argument("--mouse-btn",
                        default="middle",
                        choices=("left", "right", "middle"),
                        help="Mouse button to use as a trigger (default: middle)")

    # Enabled by default; users can turn the entire mouse trigger OFF:
    parser.add_argument("--no-mouse",
                        dest="enable_mouse_trigger",
                        action="store_false",
                        help="Disable mouse trigger completely")

    parser.add_argument("--dual-trigger-required",
                        action="store_true",
                        help="Require BOTH hot-key and mouse to be pressed")

    # Hold-to-record is now the default ✔
    parser.add_argument("--toggle-mouse-record",
                        dest="mouse_hold_to_record",
                        action="store_false",
                        help="Use press-to-start / press-again-to-stop "
                             "(disables hold-to-record)")
    parser.add_argument("--mouse-hold-threshold",
                        type=float,
                        default=0.2,
                        help="Minimum hold time (s) before recording starts")

    # Mutually exclusive clipboard options
    paste = parser.add_mutually_exclusive_group()
    paste.add_argument("--auto-paste",
                       dest="auto_paste_on_release",
                       action="store_true",
                       help="Copy text **and** press Ctrl+V on mouse release")
    paste.add_argument("--manual-paste",  # ⟸ new, user must hit Ctrl+V
                       "--copy-on-release",  # old alias still works
                       dest="copy_on_release",
                       action="store_true",
                       help="Copy only – you paste manually")

    # Back-compat alias (hidden)
    paste.add_argument("--auto-paste-on-release",
                       dest="auto_paste_on_release",
                       action="store_true",
                       help=argparse.SUPPRESS)

    # GPU optimization settings
    parser.add_argument("--torch-compile-mode", default="off",
                       choices=("off", "default", "reduce-overhead", "max-autotune"),
                       help="Mode for torch.compile() optimisation (requires PyTorch 2.1+)")
    parser.add_argument("--attention-backend", default="flash",
                       choices=("none", "flash", "mem-eff"),
                       help="Attention backend (requires PyTorch 2.0+)")
    
    # Hidden flag for CI CPU testing
    parser.add_argument("--tiny-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose - show DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Quiet   - warnings & errors only")
    
    args = parser.parse_args()
    
    if args.verbose and args.quiet:
        parser.error("choose only one of --verbose / --quiet")
    if args.verbose:
        LOGGER.setLevel("DEBUG")
    elif args.quiet:
        LOGGER.setLevel("WARNING")

    # CUDA availability guard
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("ERROR: CUDA requested but not available. Use --device cpu or install CUDA.", file=sys.stderr)
                sys.exit(1)
        except ImportError:
            print("ERROR: PyTorch not installed. Cannot use CUDA device.", file=sys.stderr)
            sys.exit(1)

    # Use tiny model for CPU testing if requested
    model_name = "tiny.en" if args.tiny_model else args.model
    
    # Auto-select appropriate compute type for device
    compute_type = args.compute
    if args.device == "cpu" and compute_type == "float16":
        compute_type = "float32"  # CPU doesn't support efficient float16
        
    # Safer defaults for new users
    initial_prompt = args.initial_prompt or ("Dictated text:" if args.language == "en" else None)
        
    # hard-clip to keep users out of the "digital clipping" zone
    mic_gain = min(args.mic_gain, 5.0)

    # --vad-aggr: turn "auto" into a sane integer default before passing to Config
    vad_aggr = 2 if args.vad_aggr == "auto" else args.vad_aggr

    # auto-quantise on compatible CPUs, etc.
    if args.compute == "auto" and args.device == "cpu":
        args.compute = "int8"
    
    # --- New, Smarter Prompt Logic ---
    final_prompt = None
    if args.initial_prompt:
        # 1. A direct --initial-prompt overrides everything.
        final_prompt = args.initial_prompt
    elif args.preset:
        # 2. If a preset is chosen, build the prompt from prompts.py
        template, terms_str = PRESETS[args.preset]
        final_prompt = template.format(terms=terms_str)

    cfg = Config(
        model_name=model_name,
        device=args.device,
        compute_type=compute_type,
        language=args.language,
        initial_prompt=final_prompt, # Pass the single, final prompt string
        beam_size=args.beam_size,
        vad_aggr=vad_aggr,
        min_confidence_threshold=args.min_confidence,
        retry_temperatures=args.retry_temperatures,
        max_buffer_seconds=args.max_buffer_s,
        input_device=args.input_device,
        mic_gain=mic_gain,
        hotkey="ctrl+space",
        # When the flag is *absent*, argparse leaves enable_mouse_trigger True
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
    )

    engine = DictationEngine(cfg)
    try:
        asyncio.run(engine.start())
        if args.bench:
            asyncio.run(engine.run_benchmark())
    except (KeyboardInterrupt, asyncio.CancelledError):
        LOGGER.info("Interrupted by user, shutting down...")
    finally:
        if 'engine' in locals():
            engine.stop()


if __name__ == "__main__":
    main()