# Dictation Performance PRD Execution Plan

## Objective

Implement GitHub issue #2: improve dictation latency, efficiency, model-stack evaluation, and architecture while preserving production reliability.

## Authority Set

- Source of truth: GitHub issue #2, "PRD: Improve dictation latency, efficiency, and model stack".
- Trusted base: `origin/master` plus the already-open release-paste fix branch state currently carried by `feature/dictation-performance-foundation`.
- Target branch: `feature/dictation-performance-foundation`.
- Conflict rule: measured behavior and tests beat stale documentation; defaults must not change without benchmark evidence.

## Scope In

- Restore benchmark/profile CLI behavior.
- Repair CLI/config/test drift discovered while implementing the PRD.
- Add a transcription backend boundary for faster-whisper model loading and inference.
- Add deterministic post-processing module and golden tests.
- Add benchmark/accuracy fixtures and model matrix command surface.
- Document measured workflow and compatibility constraints.

## Scope Out

- GUI work.
- Cloud transcription.
- Fine-tuning or training models.
- Making GPU benchmarks mandatory for the CPU unit-test gate.
- Changing the default model without evidence from the benchmark path.

## Delivery Map

One consolidation PR on `feature/dictation-performance-foundation` is approved by the user request to complete the full PRD in this pass.

Commit structure:

1. `test:` benchmark, backend, post-processing, and CLI contracts.
2. `feat:` benchmark/profile/model matrix/backend/postprocess implementation.
3. `docs:` benchmark/model guidance and compatibility notes.

Changed-code-line budget: keep human-authored implementation below 1,300 changed lines unless the full PRD requires docs/tests beyond that.

Regroup rule: if a slice requires live GPU/model downloads or cannot be verified on CPU, keep it behind a manual benchmark command and mark the evidence gap in docs rather than changing defaults.

## Affected Surface

- Changed boundary: dictation runtime measurement, model backend creation/transcription, post-processing, and CLI contract.
- Adjacent consumers: `DictationEngine`, `AudioStream`, CLI users, tests, README setup and usage guidance.
- Upstream triggers: hotkey/mouse-hold runtime, `python -m dictation_tool`, manual benchmark commands.
- No-change surfaces: single paste per release, explicit mic fail-closed behavior, CPU test speed, separate GPU smoke validation.

## Verification Gates

- Focused CPU tests for touched modules.
- `black --check` on touched Python files.
- `ruff check` on touched Python files.
- Best-effort non-GPU suite; if it hangs or fails due unrelated stale tests, capture blocker honestly.
- GPU/model matrix remains manual unless the local environment can run it safely.

## Execution Checklist

- [x] Restore benchmark/profile CLI contract and current engine tests.
- [x] Fix no-VAD ring initialization/full-buffer regression.
- [x] Create governing execution plan.
- [x] Add transcription backend adapter and tests.
- [x] Extract deterministic post-processing module with golden tests.
- [x] Add benchmark fixture module and model matrix CLI surface.
- [x] Add accuracy fixture schema and tests that do not require GPU by default.
- [x] Update README/CHANGELOG for benchmark/model guidance and process cleanup.
- [x] Run focused verification and record any full-suite blockers.
- [x] Install the latest supported faster-whisper runtime in the active venv.
- [x] Live-benchmark `large-v3`, `medium.en`, `large-v3-turbo`, and `distil-large-v3`.
- [x] Decide the default/recommended model profile from measured latency and transcript quality.
- [x] Reconcile VAD ownership (`webrtcvad` vs faster-whisper `vad_filter`) from measured results.

## PRD Completion Status

The PRD live-evaluation gate is complete on the target Windows RTX 3080 runtime.
The default model is changed to `large-v3-turbo` for English dictation: the
measured matrix showed it is close to `distil-large-v3` latency, and live user
dictation feedback showed better accuracy than `distil-large-v3`.

Current evidence:

- Active `.venv310` has `faster-whisper 1.2.1`.
- Active `.venv310` has `ctranslate2 4.7.1`.
- Local GPU is available: NVIDIA GeForce RTX 3080, driver 591.74, CUDA 13.1
  driver capability, PyTorch `2.5.1+cu121`.
- Upstream faster-whisper `1.2.1` supports modern Whisper-family evaluation,
  including `distil-large-v3` and `large-v3-turbo` compatible model paths.
- The app now keeps faster-whisper model load, inference, and close operations on
  one dedicated model thread; this resolved the native Windows CUDA/CTranslate2
  teardown crash found during the live matrix.

Live matrix results:

- Generated 11.782 s speech fixture, model VAD enabled: `distil-large-v3`
  p50 356.503 ms, p95 360.189 ms, RTF 33.074x, WER 0.16.
- Generated 11.782 s speech fixture, model VAD enabled: `large-v3-turbo`
  p50 413.929 ms, p95 420.441 ms, RTF 28.354x, WER 0.24.
- Generated 11.782 s speech fixture, model VAD enabled: `medium.en`
  p50 616.077 ms, p95 695.492 ms, RTF 18.384x, WER 0.28.
- Generated 11.782 s speech fixture, model VAD enabled: `large-v3`
  p50 953.087 ms, p95 955.916 ms, RTF 12.374x, WER 0.24.
- Public JFK 11.000 s human speech fixture, model VAD disabled:
  `distil-large-v3` p50 320.776 ms, p95 332.155 ms, RTF 33.913x, WER 0.1364.
- Public JFK 11.000 s human speech fixture, model VAD disabled:
  `large-v3-turbo` p50 345.234 ms, p95 350.654 ms, RTF 31.743x, WER 0.1364.
- Public JFK 11.000 s human speech fixture, model VAD disabled:
  `large-v3` p50 779.563 ms, p95 779.955 ms, RTF 14.144x, WER 0.1364.
- Public JFK 11.000 s human speech fixture, model VAD disabled:
  `medium.en` p50 490.587 ms, p95 500.177 ms, RTF 22.358x, WER 0.1364.
- Live dictation follow-up: `distil-large-v3` felt less accurate in real use,
  while `large-v3-turbo` felt materially better and still very fast. This
  supersedes the fixture-only default recommendation.
- VAD comparison on the generated speech fixture showed disabling
  faster-whisper internal VAD slightly improved latency for top candidates
  without changing WER, so WebRTC VAD remains the live microphone segmentation
  owner and `--no-model-vad` is available for benchmark/runtime comparison.

## Verification Notes

- `pytest tests -q -m "not gpu"` passes: 109 passed, 3 deselected.
- `pytest tests/test_gpu_smoke.py -q -m gpu` passes: 3 passed.
- Focused PRD tests pass across CLI, engine, audio I/O, backend, post-processing, benchmark helpers, config, and VAD.
- `black --check dictation_tool tests` passes.
- `ruff check dictation_tool pyproject.toml` passes.
- `mypy dictation_tool` passes.
- `pip check` passes.
