# 📝 Dictation‑Tool — Faster‑Whisper CLI

> **Hold your mouse button, speak, release, and watch text materialise wherever your cursor is.**

A lightning‑fast desktop dictation utility for **Windows 10/11** (Linux & macOS untested) powered by [`faster‑whisper`](https://github.com/SYSTRAN/faster-whisper) and accelerated by **Flash‑Attention 2**.

<p align="center">
  <img src="docs/demo.gif" width="640" alt="Dictation‑Tool hold‑to‑talk demo">
</p>

---

## ✨ Features

|                            |                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| **Instant hold‑to‑record** | Press a mouse button (configurable, default: right) for ≥ 0.2 s, speak, release to paste. |
| **Ultra‑low latency**      | CUDA 12 + Flash‑Attention kernels & zero‑copy audio pipeline.                             |
| **Smart VAD**              | Real‑time segmentation with tolerant fallback to avoid truncation.                        |
| **Context memory**         | Remembers recent sentences for better proper‑noun accuracy.                               |
| **Clipboard modes**        | Auto‑paste, copy‑only, or clipboard‑off.                                                  |
| **CLI everything**         | 25+ flags: mic gain, device, model size, beam width, VAD aggressiveness, etc.             |

---

## 📥 Quick install

> **Prerequisites**
> • **Python 3.10 (64‑bit)** — other versions untested.
> • **NVIDIA GPU + driver ≥ 545** (CUDA 12 runtime).
> • \~2 GB free disk space (first model download).

```bash
# 1) Clone & enter the repo
git clone https://github.com/yourname/dictation-tool.git
cd dictation-tool

# 2) Create / activate a virtual env
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3) Install runtime deps (PyTorch wheel index embedded)
pip install -r requirements.txt
```

*Developer tooling* (`pytest`, `ruff`, etc.) lives in `requirements-dev.txt`.

---

## 🚀 Quick start & mic tuning

1. **Tune microphone gain**
   Start verbose mode to view RMS levels:

```bash
python -m dictation_tool -v --auto-paste --mic-gain 2.5
```

Speak and watch the RMS in logs — target **‑25 dBFS to ‑15 dBFS**.
• RMS too low (e.g., ‑35 dBFS) → raise gain: `--mic-gain 3.5`
• RMS too high (e.g., ‑5 dBFS)  → lower gain: `--mic-gain 1.5`

2. **Pick your performance mode**

### 🎯 Option A — Maximum accuracy

```bash
python -m dictation_tool --model large-v3 --auto-paste --mic-gain <your_gain> --vad-aggr 2
```

Requires a mid‑tier GPU (RTX 30‑series+) and adds \~1‑2 s initial latency.

### 💨 Option B — Maximum speed (with prompt tricks)

Use the lightweight `medium.en` model and regain accuracy via prompts.

*Preset examples*

```bash
# Business e‑mails / docs
python -m dictation_tool --model medium.en --preset business --auto-paste

# Programming / code review
python -m dictation_tool --model medium.en --preset programming --auto-paste
```

*Fully custom prompt*

```bash
python -m dictation_tool --model medium.en \
  --initial-prompt "Whisper, VAD, CUDA, PyTorch, Dictation‑Tool"
```

*Increase beam size*

```bash
python -m dictation_tool --model medium.en --preset general --beam-size 5
```

Adds a small latency bump but improves word choice.

---

## 🛠 Common customisations

| Goal                            | Flag             | Example                                  |
| ------------------------------- | ---------------- | ---------------------------------------- |
| Disable VAD for long monologues | `--no-vad`       | `--no-vad --max-buffer-seconds 30`       |
| Tolerate longer pauses          | `--vad-aggr`     | `--vad-aggr 1` (0 =tolerant … 3 =strict) |
| Copy without pasting            | `--manual-paste` | `... --manual-paste`                     |
| Change mouse trigger            | `--mouse-btn`    | `--mouse-btn middle`                     |
| Use hotkey only (no mouse)      | `--no-mouse`     | `... --no-mouse`                         |

Run `python -m dictation_tool --help` for the full flag list.

---

## 🌳 Project layout

```text
dictation_tool/
├─ __main__.py     # CLI entry‑point
├─ engine.py       # DictationEngine core
├─ io.py           # Audio + VAD helpers
└─ …               # More modules
 tests/
 docs/
```

---

## ⚙️ System requirements

| Component    | Requirement                                  |
| ------------ | -------------------------------------------- |
| Python       | **3.10 x64**                                 |
| CUDA runtime | 12.1+ (driver ≥ 545)                         |
| GPU          | NVIDIA RTX (≥ 8 GB VRAM recommended)         |
| OS           | Windows 10/11 (Linux/macOS expected to work) |

A CPU‑only run is possible but latency rises > 5×.

---

## 📄 License & contributing

MIT — see **LICENSE**.
PRs and issues welcome; please run `ruff` and `pytest -n auto` before pushing.
