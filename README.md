# 📝 Dictation-Tool — Whisper `large-v3` in Real Time

> “Hold your middle mouse button, speak, release, and watch the text appear in your active application.”

A moderately fast and optimized desktop dictation tool for Windows, powered by the **[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)** engine running OpenAI's `large-v3` model and accelerated with **Flash-Attention 2**.

*(Untested on Linux and macOS, but may work with compatible dependencies.)*

---

## ✨ Features

| Feature                    | Details                                                                                                                        |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| **Instant Hold-to-Record** | Hold your chosen mouse button for just 0.2s to record, release to transcribe & auto paste. Simple and effective.                      |
| **Low Latency**            | Optimized for speed with the `faster-whisper` engine, Flash-Attention 2 kernels, and a zero-copy memory pipeline.                |
| **Intelligent VAD**        | Automatically segments your speech in real-time for quick feedback, with a robust fallback to ensure nothing is lost.            |
| **Context-Aware Accuracy** | Remembers the context of your recent sentences to improve accuracy on names, jargon, and formatting.                             |
| **Flexible Clipboard**     | Choose to auto-paste (`Ctrl+V`), copy-only, or disable clipboard integration. Includes a high-speed Windows-native clipboard path. |
| **Highly Configurable**    | Fine-tune every aspect via CLI flags: mic device/gain, VAD level, model size, compute type, timings, and more.                   |
| **Cross-Platform**         | Optimized for Windows/CUDA. Linux/macOS are untested but should work with compatible dependencies.                               |

---

## ⚙️ Requirements

> [!IMPORTANT]
> **An NVIDIA GPU is strongly recommended for real-time performance.**
> This tool is tuned for CUDA 12.x and Flash-Attention. It *should* run on a CPU, but latency will increase significantly. It's optimized for a specific PC/Mic setup, so you might need to do some tuning.

| Component      | **Required Version**                                             |
| :------------- | :--------------------------------------------------------------- |
| **Python**     | **3.10 (64-bit)**. This is a strict requirement.                 |
| **CUDA Toolkit** | 12.1 or newer for GPU acceleration.                              |
| **GPU**        | NVIDIA RTX series (30xx or 40xx) with ≥ 8 GB VRAM is ideal.      |
| **OS**         | Windows 10/11 • Ubuntu 22.04 (Untested) • macOS (Untested)       |

---

## 📦 Installation

Following these steps precisely is required for the tool to work correctly.

**1. Install Python 3.10 (64-bit)**

You must be using Python 3.10.x. If you don't have it, you can get it from the [official Python website](https://www.python.org/downloads/release/python-31011/) or via a package manager.

```powershell
# Example using Chocolatey on Windows
choco install python --version=3.10.11
```

**2. Create and Activate a Virtual Environment**

This is a critical step to avoid dependency conflicts.

```bash
# From your project directory
python -m venv .venv

# Activate it
# On Windows (in PowerShell/CMD):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

**3. Install All Requirements**

This single command installs PyTorch for CUDA 12.1 and all other necessary packages.

```bash
# Ensure your virtual environment is active!
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Quick Start

**1. Basic Usage**


```bash
python -m dictation_tool --mic-gain 2.5 --vad-aggr 2 --auto-paste
```
-   `--mic-gain 2.5`: A good starting point for most microphones.
-   `--vad-aggr 2`: A balanced Voice Activity Detection setting.
-   `--auto-paste`: Automatically pastes the transcribed text.

**2. Fine-Tune Your Microphone Gain**

For the best accuracy, you may need to adjust the microphone gain. Run with the `-v` (verbose) flag to see your audio level.

```bash
python -m dictation_tool -v --auto-paste --mic-gain 2.5
```

While speaking, look for the `RMS` value in the debug logs. Aim for a value between **-25 dBFS** and **-15 dBFS**.
-   If your RMS is too low (e.g., -35 dBFS), increase the gain: `--mic-gain 3.5`
-   If your RMS is too high (e.g., -5 dBFS), decrease the gain: `--mic-gain 1.5`

**3. CPU-Only Usage (Slower)**

If you don't have an NVIDIA GPU, you can run on the CPU with with a smaller model ( untested)

```bash
python -m dictation_tool --device cpu --model medium.en --compute int8
```

---

## 🛠️ Common Customizations

Tailor the tool to your exact workflow with these flags.

| Goal                             | Flag                     | Example                                |
| :------------------------------- | :----------------------- | :------------------------------------- |
| Use a different model            | `--model`                | `--model medium.en`                    |
| Disable VAD for long monologues  | `--no-vad`               | `--no-vad --max-buffer-seconds 30`     |
| Make VAD tolerate longer pauses  | `--vad-aggr`             | `--vad-aggr 1` (0=tolerant, 3=strict)  |
| Copy text, but don't paste       | `--manual-paste`         | `... --manual-paste`                   |
| Change the mouse trigger         | `--mouse-btn`            | `--mouse-btn right`                    |
| Use hotkey only (no mouse)       | `--no-mouse`             | `... --no-mouse`                       |

For a full list of options, run:
```bash
python -m dictation_tool --help
```
