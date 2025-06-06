# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GPU smoke tests for CUDA functionality validation
- Comprehensive latency benchmarking script with memory tracking
- Input device selection via `--input-device` flag
- Microphone gain control via `--mic-gain` flag
- `clamp_tokens()` utility for reasoning token budget management
- CUDA availability guard at startup
- Safe audio gain scaling with clipping to prevent overflow

### Changed
- **BREAKING**: Unified attention backend configuration into single `--attention-backend` enum (none, flash, mem-eff)
- **BREAKING**: Removed legacy `--no-flash` and `--no-mem-efficient` flags
- Improved clipboard operations with retry wrapper for reliability
- Enhanced retry logic with proper temperature slicing and avg_logprob validation
- Audio gain processing now uses safe int16 clipping
- Test timeouts added for GPU operations
- Version bumped to 1.1.0-dev

### Fixed
- GPU optimization model compilation now accepts kwargs correctly
- Memory-efficient attention backend selection
- Retry temperature enforcement respects max_retries limit
- Audio input device selection and gain application

### Performance
- 5-10% latency improvement from architectural optimizations
- GPU-first configuration defaults
- Sub-800ms transcription target compliance

## [1.0.0] - Previous Release
- Initial production release with basic dictation functionality
