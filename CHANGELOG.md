# Changelog
All notable changes to **dictation-tool** are recorded in this file.  
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- **GPU smoke tests** – verify CUDA availability at startup.  
- **Comprehensive latency script** – track wall-time and memory.  
- **Input device selection** – `--input-device`.  
- **Microphone gain control** – `--mic-gain` with safe clipping.  
- **Token-budget utility** – `clamp_tokens()` for reasoning prompts.  
- **CUDA guard** – exit early when no GPU detected.

### Changed
- **BREAKING** – unified attention backend: `--attention-backend {none,flash,mem-eff}`; removed `--no-flash` and `--no-mem-efficient`.  
- Upgraded clipboard operations with retry wrapper.  
- Improved retry logic (temperature slicing, `avg_logprob` validation).  
- Refactored audio-gain code to use int16 clipping.  
- Added GPU test timeouts.

### Fixed
- Passed kwargs correctly to GPU optimisation compiler.  
- Resolved memory-efficient attention selection edge-case.  
- Enforced `max_retries` limit in retry logic.  
- Applied input-device selection and gain consistently.

### Performance
- Achieved 5–10 % latency reduction through architectural tweaks.  
- Defaulted to GPU-first configuration.  
- Met sub-800 ms transcription target.

## [1.2.0] – 2025-06-08

### Added
* **Mouse-hold long-dictation support** – unlimited shadow buffer captures up to `max_buffer_seconds` (now 60 s by default).  
* **Name/e-mail accuracy helpers**  
  * Regex post-processor converts "_foo at bar dot com_" → `foo@bar.com`.  
  * `initial_prompt` is now honoured continuously for rare names (e.g. *McKenzie*).  
* **Punctuation map** accepts "at sign" & "dot" for `@` / `.`.  
* Config default `max_buffer_seconds` raised to **60 s**.

### Changed
* **BREAKING** – removed experimental sample-counter shadow logic  
  * Deleted `self._shadow_samples` & `self._shadow_cap`.  
  * `_add_to_shadow` is now a single `deque.append`.  
* Mouse-hold path resets shadow only at start/end of a hold; streaming mode still clears after every VAD flush.  
* Clipboard retry now uses a single thread-pool executor for robustness.

### Fixed
* 10 s truncation bug when devices delivered jumbo audio chunks.  
* Rare "deque mutated during iteration" race removed by dropping bounded-deque logic.  
* Correctly clears shadow buffer counter in streaming VAD path.

### Performance
* Simplified shadow handling saves ~3 µs per chunk and ~0.5 MB RAM.  
* No impact on Whisper logits – acoustic fidelity unchanged.

---

## [1.1.0] – 2025-06-07

### Added
- **Configurable accuracy** – `--beam-size` for quality vs speed.  
- **Advanced prompting** – `--preset` and `--initial-prompt` for style and vocabulary control.  
- **Smart text post-processing** – fixes spacing in URLs/paths (e.g. `www . google . com` → `www.google.com`).  
- **On-demand latency benchmarking** – `--bench` flag.

### Fixed
- Corrected mutable default for `prompt_terms` in `Config`.  
- Ensured `engine.stop()` executes on exit.  
- Made benchmark run safely inside existing asyncio loop.

## [1.0.0] – 2025-06-06
### Added
- Initial production release providing baseline dictation functionality.
