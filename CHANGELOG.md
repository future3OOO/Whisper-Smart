# Changelog
All notable changes to **dictation-tool** are recorded in this file.  
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
the project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]
*(no changes yet)*


## [1.3.0] – 2025-06-09

### Added
- **“Email” preset** – turnkey prompt for business mail; pairs with new
  command-cue engine (see README).
- **Command cues** recognised in speech  
  * `new line / line break / newline` → `\n`  
  * `new paragraph` → double `\n\n` with automatic full-stop before break  
  * `bullet point` → `\n• ` (unicode bullet)  
- **Post-processor utilities**  
  * Regex normaliser collapses **Unicode & ASCII spaces** around `@` and `.`.  
  * Duplicate-comma squasher + comma-to-period upgrade before blank lines.  
  * Safety pass removes residual smart quotes, back-ticks, or `�` (U+FFFD).  
- **Sign-off normaliser** – capitalises “kind regards / best regards / …” and
  ensures a trailing comma, then inserts the missing newline.
- **Line-capitaliser** – first alphabetic character on every line is
  upper-cased (handles bullets, tabs, spaces).

### Changed
- `_EMAIL_RE` extended to match both *“at gmail dot com”* and
  *“at **sign** gmail dot com”*.
- `_CMD_SUBS` now consumes any optional punctuation that immediately follows a
  cue (e.g. “new paragraph,” or “new line.”) so stray commas/periods no longer
  leak into the text.
- README rewritten with **PowerShell-style commands**, new medium-model e-mail
  example, beam-size notes, and bullet-point demo.
- Default `max_buffer_seconds` remains 60 s but shadow buffer is truly
  unlimited during mouse-hold; cleared only on flush.

### Fixed
- New-line swallowing bug after sign-off normalisation.  
- “Comma storm” when the word *comma* plus punctuation produced `,,,`.  
- Address formatter no longer removes new-lines when trimming spaces.  
- Capitalisation now honours sentences that start after a bullet or tab.

### Performance
- Regexes compiled at module load; zero per-utterance overhead.  
- Single thread-pool executor handles all clipboard writes; avoids thread spam.  
- No measurable slowdown: median inference latency unchanged on RTX 3080.


## [1.2.0] – 2025-06-08

### Added
- **Mouse-hold long-dictation support** – unlimited shadow buffer captures up to
  `max_buffer_seconds` (default 60 s).  
- **Name/e-mail accuracy helpers** – regex post-processor converts *foo at bar
  dot com* → `foo@bar.com`.  
- **Punctuation map** accepts “at sign” & “dot” for `@ / .`.  
- Config default `max_buffer_seconds` raised to **60 s**.

### Changed
- **BREAKING** – removed experimental sample-counter shadow logic.  
- Mouse-hold path resets shadow only at start/end of a hold; streaming mode
  still clears after each VAD flush.  
- Clipboard retry now uses a single thread-pool executor.

### Fixed
- 10 s truncation bug when drivers delivered jumbo audio chunks.  
- “Deque mutated during iteration” race removed by dropping bounded-deque logic.

### Performance
- Simplified shadow handling saves ≈ 3 µs per chunk and ≈ 0.5 MB RAM.


## [1.1.0] – 2025-06-07

### Added
- **Configurable accuracy** – `--beam-size` for quality vs. speed.  
- **Advanced prompting** – `--preset` and `--initial-prompt`.  
- **Smart text post-processing** – fixes spacing in URLs/paths
  (`www . google . com` → `www.google.com`).  
- **On-demand latency benchmarking** – `--bench` flag.

### Fixed
- Corrected mutable default for `prompt_terms` in `Config`.  
- Ensured `engine.stop()` executes on exit.  
- Made benchmark run safely inside existing asyncio loop.


## [1.0.0] – 2025-06-06
### Added
- Initial production release providing baseline dictation functionality.
