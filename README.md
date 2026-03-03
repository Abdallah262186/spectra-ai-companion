# Spectra 2.0 – Personal AI Companion

A fully local, privacy-first AI companion that runs on modest hardware,
learns from your conversations over time, monitors PC activities, and
initiates proactive conversations.

---

## Features

- **Local inference** – Qwen2-1.5B-Instruct loaded with 8-bit quantisation
  uses ≤ 2 GB VRAM so gaming on the same GPU is still comfortable.
- **LoRA fine-tuning** – Spectra trains a personal adapter on your own
  conversation history every 50 exchanges (configurable).
- **PC activity awareness** – monitors Spotify playback, browser tabs,
  Downloads folder, and running processes.
- **Proactive conversations** – Spectra can start a conversation based on
  what you are doing (respects quiet hours).
- **Web search** – `/search <query>` fetches DuckDuckGo results offline-first.
- **Persistent memory** – SQLite database with full-text search.
- **Polished terminal UI** – coloured output via colorama + rich.

---

## Hardware Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU | NVIDIA GTX 1660 Super (6 GB VRAM) | Tested on GTX 1660 Super; any CUDA-capable GPU with ≥ 6 GB VRAM should work. Do **not** use the GT 1030 (insufficient VRAM) |
| CPU | Intel Xeon E5-1650 v4 (6c/12t) | Any modern x86 CPU works |
| RAM | 16 GB | 32 GB recommended |
| Storage | 5 GB free | Model download ~3 GB |
| OS | Windows 10 / 11 | Monitoring modules are Windows-only |

> **VRAM budget:** the model uses ≈ 2 GB in 8-bit mode, leaving ≥ 4 GB free
> for gaming on the GTX 1660 Super.

---

## Installation

### 1. Prerequisites

- Python 3.10 or 3.11 (64-bit)
- NVIDIA driver ≥ 528 and CUDA 11.8 or 12.x
- `pip` (comes with Python)

### 2. Clone the repository

```bash
git clone https://github.com/Abdallah262186/spectra-ai-companion.git
cd spectra-ai-companion
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 4. Install PyTorch with CUDA

Visit https://pytorch.org/get-started/locally/ and copy the install command
for your CUDA version.  For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` requires the MSVC Build Tools on Windows.
> Download them from https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 6. First run

```bash
python main.py
```

On first launch Spectra will:
1. Download the Qwen2-1.5B-Instruct model (~3 GB) from HuggingFace.
2. Scan your Documents, Downloads, and Music folders.
3. Read the Windows registry for installed programs.
4. Parse Opera GX / Chrome bookmarks.
5. Start the interactive chat loop.

---

## Usage

### Starting Spectra

```bash
python main.py                 # Normal startup
python main.py --skip-scan     # Skip the initial PC scan
python main.py --no-proactive  # Disable proactive conversations
python main.py --train-now     # Run LoRA training then enter chat
python main.py --config custom.yaml  # Use a different config file
```

### Chat commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/quit` | Exit Spectra gracefully |
| `/status` | Show model and session information |
| `/train` | Manually trigger LoRA fine-tuning |
| `/search <query>` | Search DuckDuckGo |
| `/scan` | Re-run the PC scanner |
| `/memory` | Display recent conversation history |

---

## Configuration

All settings live in `config.yaml`.  The most important ones:

```yaml
model:
  name: "Qwen/Qwen2-1.5B-Instruct"
  quantization: "8bit"     # Keep this to stay within 2 GB VRAM
  max_new_tokens: 150      # Increase for longer responses (uses more VRAM)
  temperature: 0.7

companion:
  name: "Spectra"
  quiet_hours_start: 23    # No proactive messages 23:00–08:00
  quiet_hours_end: 8

training:
  trigger_every_n_conversations: 50   # Auto-train after N turns
  lora_rank: 8             # Higher = more capacity, more VRAM
```

---

## Training

Spectra fine-tunes itself on your conversation history using QLoRA:

- Training is triggered automatically every 50 conversations.
- You can also trigger it manually with `/train` or `--train-now`.
- Adapters are saved with timestamps in the `adapters/` directory.
- The latest adapter is loaded automatically on the next startup.

Training on a GTX 1660 Super with ~50 conversation pairs takes roughly
5–15 minutes depending on response length.

---

## Troubleshooting

### "CUDA out of memory"
- Lower `max_new_tokens` in `config.yaml` (try 100 or 80).
- Increase `gradient_accumulation_steps` in `lora_trainer.py` and reduce
  batch size.
- Close other GPU-intensive applications before training.

### "Model download is slow"
- The model is ~3 GB; it is only downloaded once.
- Set the environment variable `HF_HOME` to a fast drive.

### "bitsandbytes not working on Windows"
- Make sure you have the MSVC Build Tools installed.
- Try `pip install bitsandbytes --prefer-binary`.

### Monitoring shows no Spotify data
- Spectra reads the Spotify window title, which requires Spotify desktop
  (not the web player).
- `pygetwindow` and `win32gui` must be installed.

### Proactive messages not appearing
- Check that `proactive.enabled: true` is set in `config.yaml`.
- Ensure the current time is outside `quiet_hours_start/end`.

---

## Project Structure

```
spectra-ai-companion/
├── main.py                  # Entry point
├── config.yaml              # All settings
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── adapters/                # Saved LoRA adapter checkpoints
├── spectra/
│   ├── core/
│   │   ├── engine.py        # Model loading & inference
│   │   ├── conversation.py  # Chat loop & commands
│   │   └── proactive.py     # Background proactive thread
│   ├── memory/
│   │   ├── database.py      # SQLite persistence
│   │   └── context.py       # Dynamic prompt assembly
│   ├── monitoring/
│   │   ├── spotify.py       # Spotify track detection
│   │   ├── browser.py       # Browser activity
│   │   ├── downloads.py     # Downloads folder watcher
│   │   ├── processes.py     # Running process tracker
│   │   └── scanner.py       # Initial PC scan
│   ├── training/
│   │   ├── data_prep.py     # Conversation → JSONL
│   │   └── lora_trainer.py  # QLoRA fine-tuning
│   └── search/
│       └── web_search.py    # DuckDuckGo search
```

---

## Roadmap

- [ ] GUI overlay (system tray icon + pop-up chat)
- [ ] Voice input via Whisper
- [ ] Voice output via local TTS (Coqui / Piper)
- [ ] Larger context window support (Qwen2-7B when VRAM allows)
- [ ] Plugin system for custom monitoring modules
- [ ] Scheduled summaries ("Here's what I noticed today…")

---

## License

MIT – free to use, modify, and distribute.