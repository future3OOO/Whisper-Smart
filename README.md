# 📝 Dictation-Tool — Faster-Whisper CLI

> **Hold your mouse button, speak, release, and watch text materialise wherever your cursor is.**

A lightning-fast desktop dictation utility for **Windows 10/11** (Linux & macOS untested) powered by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) and accelerated by **Flash-Attention 2**.

<p align="center">
  <img src="docs/demo.gif" width="640" alt="Dictation-Tool hold-to-talk demo">
</p>

---

## ✨ Features

|                              |                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **Instant hold-to-record**   | Press the chosen mouse button (default = right) for ≥ 0.2 s, speak, release to paste.         |
| **Ultra-low latency**        | CUDA 12 + Flash-Attention 2 kernels & zero-copy audio pipeline.                                |
| **Smart VAD**                | Real-time segmentation with tolerant fallback—no lost syllables.                              |
| **Context memory**           | Remembers recent sentences for better proper-noun accuracy.                                   |
| **Clipboard modes**          | Auto-paste, copy-only, or clipboard-off.                                                      |
| **CLI everything**           | 25 + flags: mic gain, device, model size, beam width, VAD aggressiveness …                    |

---

## 📥 Quick install

> **Prerequisites**  
> • **Python 3.10 (64-bit)** — other versions untested  
> • **NVIDIA GPU + driver ≥ 545** (CUDA 12 runtime)  
> • ≈ 2 GB free disk space (first model download)

```powershell
# 1) Clone & enter the repo
git clone https://github.com/yourname/dictation-tool.git
cd dictation-tool

# 2) Create / activate a virtual env
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

# 3) Install runtime deps (PyTorch wheel index embedded)
pip install -r requirements.txt
```

Developer tooling (pytest, ruff, etc.) lives in requirements-dev.txt.

## 🚀 Quick start & mic tuning

### 1 Tune microphone gain

```powershell
python -m dictation_tool -v --auto-paste --mic-gain 2.5
```

In verbose logs aim for **-25 dBFS … -15 dBFS**  
 • Raise `--mic-gain` if RMS ≈ -35 dBFS  • Lower if RMS ≈ -5 dBFS

### 2 Choose a performance mode

#### 🎯 Option A — Maximum accuracy (large-v3)

```powershell
python -m dictation_tool `
       --model large-v3 `
       --auto-paste `
       --mic-gain <your_gain> `
       --vad-aggr 2
```

Latency ≈ 200-500 ms on an RTX 3080. Designed for 20-30 s dictation bursts.

#### 💨 Option B — Maximum speed (medium.en + prompt tricks)

medium.en delivers ≈ 5-20 ms interface latency while staying surprisingly
accurate when paired with a good prompt and a larger beam.

## 📧 Fast e-mail workflow — preset **email**

```powershell
python -m dictation_tool `
       --model medium.en `
       --preset email `
       --beam-size 5 `
       --auto-paste
```

<details>
<summary>Example conversation ▶️</summary>

```
🎙️  SPOKEN
-------------------------------------------
hi Steve new paragraph
how's your day going i hope everything is well new paragraph
please email john@gmail.com new paragraph
kind regards new line
john
bullet point first item
bullet point second item

📋  CLIPBOARD
-------------------------------------------
Hi Steve,

How's your day going? I hope everything is well.

Please email john@gmail.com.

Kind regards,
John
• first item
• second item
```

</details>

**What happened?**

✔ Converts "at sign / dot com" → @gmail.com  

✔ Inserts a comma after greeting, a full-stop before blank lines  

✔ Normalises sign-offs & capitalises every new line  

✔ Recognises bullet point cue (unicode bullet)

## Other presets

```powershell
# Business reports
python -m dictation_tool --model medium.en --preset business --auto-paste

# Programming / code review
python -m dictation_tool --model medium.en --preset programming --auto-paste
```

## Fully custom prompt

```powershell
python -m dictation_tool `
       --model medium.en `
       --initial-prompt "Support ticket, error code, patch, deployment" `
       --beam-size 5 `
       --auto-paste
```

Larger beam sizes (e.g. 6-8) are supported but add latency.

## 🛠 Common customisations

| Goal | Flag | Example (PowerShell) |
|------|------|---------------------|
| Disable VAD for long monologues | `--no-vad` | ... --no-vad --max-buffer-seconds 30 |
| Tolerate longer pauses | `--vad-aggr` | --vad-aggr 1 (0 =tolerant … 3 =strict) |
| Copy without pasting | `--manual-paste` | ... --manual-paste |
| Change mouse trigger | `--mouse-btn` | --mouse-btn middle |
| Use hotkey only (no mouse) | `--no-mouse` | ... --no-mouse |

Full flag list:

```powershell
python -m dictation_tool --help
```

## 🌳 Project layout

```text
dictation_tool/
├─ __main__.py     # CLI entry-point
├─ engine.py       # DictationEngine core
├─ io.py           # Audio + VAD helpers
└─ …               # More modules
 tests/
 docs/
```

## ⚙️ System requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 x64 |
| CUDA runtime | 12.1 + (driver ≥ 545) |
| GPU | NVIDIA RTX (≥ 8 GB VRAM recommended) |
| OS | Windows 10/11 (Linux/macOS should work) |

CPU-only runs are possible but ≈ 5 × slower.

## 📄 License & contributing

MIT — see LICENSE.  
PRs & issues welcome; please run ruff and pytest -n auto before pushing.
