# Spectra 2.0 — Personal AI Companion

A locally-running, privacy-first AI companion that learns from you over time
through LoRA fine-tuning, monitors your PC activity, and initiates
proactive conversations — all without sending a single byte to the cloud.

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Local LLM** | Qwen2-1.5B-Instruct loaded in 8-bit (≤ 2 GB VRAM) |
| 💬 **Interactive REPL** | Coloured terminal chat with slash-commands |
| 🎵 **Spotify awareness** | Detects what you are listening to |
| 🌐 **Browser awareness** | Detects streaming services and active tabs |
| 📁 **Downloads watcher** | Logs new files in your Downloads folder |
| 🖥️ **Process monitor** | Knows when you are gaming, coding, etc. |
| 🔍 **Web search** | DuckDuckGo search with `/search <query>` |
| 🏋️ **LoRA fine-tuning** | Auto-trains on your conversation history |
| 🔔 **Proactive system** | Sends context-aware messages at random intervals |
| 🗄️ **Persistent memory** | SQLite database — no cloud, no subscriptions |

---

## Hardware Requirements

| Component | Minimum | Used in this project |
|-----------|---------|----------------------|
| GPU | NVIDIA GTX 1660 Super (6 GB VRAM) | ✅ |
| CPU | Intel Core i5 (4 cores) | Intel Xeon E5-1650 v4 (6c/12t) ✅ |
| RAM | 16 GB | 32 GB DDR4 ✅ |
| Storage | 10 GB free | 500 GB free ✅ |
| OS | Windows 10/11 | Windows ✅ |

> **Important:** The model is loaded with 8-bit quantization and consumes
> approximately 2 GB VRAM, leaving 4 GB free for gaming.

---

## Project Structure

```
spectra-ai-companion/
├── main.py                       # Entry point — starts Spectra
├── config.yaml                   # All configurable settings
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── spectra/
    ├── core/
    │   ├── engine.py             # AI engine: model loading & inference
    │   ├── conversation.py       # Interactive REPL & slash-commands
    │   └── proactive.py          # Background proactive message system
    ├── memory/
    │   ├── database.py           # SQLite persistence layer
    │   └── context.py            # Dynamic system-prompt assembler
    ├── monitoring/
    │   ├── manager.py            # Monitoring coordinator
    │   ├── spotify.py            # Spotify playback detection
    │   ├── browser.py            # Browser / streaming detection
    │   ├── downloads.py          # Downloads folder watcher
    │   ├── processes.py          # Running process categorisation
    │   └── scanner.py            # One-time PC profile scanner
    ├── training/
    │   ├── data_prep.py          # Conversation → JSONL conversion
    │   └── lora_trainer.py       # QLoRA fine-tuning with PEFT + TRL
    └── search/
        └── web_search.py         # DuckDuckGo search wrapper
```

---

## Installation

### 1. Prerequisites

- Python 3.10 or 3.11 (recommended)
- NVIDIA GPU with CUDA 11.8+ drivers installed
- Git

### 2. Clone the repository

```bash
git clone https://github.com/Abdallah262186/spectra-ai-companion.git
cd spectra-ai-companion
```

### 3. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 4. Install PyTorch with CUDA support

Visit https://pytorch.org/get-started/locally/ and pick the right command for
your CUDA version. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the model

The model is downloaded automatically from HuggingFace on first run. You need
approximately 3 GB of free storage and an internet connection for the initial
download. After that, everything runs offline.

---

## Usage

### Start Spectra

```bash
python main.py
```

### Command-line options

| Flag | Description |
|------|-------------|
| `--skip-scan` | Skip the initial PC profile scan |
| `--no-proactive` | Disable background proactive messages |
| `--train-now` | Run LoRA fine-tuning immediately after startup |
| `--config <path>` | Use an alternative config file |

### Chat commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/quit` | Exit Spectra |
| `/status` | Show model info and current monitoring status |
| `/memory` | Show the 10 most recent conversation turns |
| `/search <query>` | Search the web via DuckDuckGo |
| `/scan` | Re-run the PC profile scan |
| `/train` | Manually trigger LoRA fine-tuning |

---

## Configuration

Edit `config.yaml` to customise Spectra's behaviour.

```yaml
model:
  name: "Qwen/Qwen2-1.5B-Instruct"   # HuggingFace model ID
  quantization: "8bit"                 # "8bit" or "none"
  max_new_tokens: 150                  # Max response length
  temperature: 0.7                     # Creativity (0 = deterministic)
  device: "cuda:0"                     # GPU device

companion:
  name: "Spectra"                      # Display name
  personality: "friendly, curious, witty, concise"
  quiet_hours_start: 23                # No proactive messages after 23:00
  quiet_hours_end: 8                   # Resume after 08:00

proactive:
  enabled: true
  min_interval_minutes: 30
  max_interval_minutes: 120

training:
  enabled: true
  trigger_every_n_conversations: 50   # Auto-prompt after N turns
  lora_rank: 8
  lora_alpha: 16
  learning_rate: 0.0002
  epochs: 3
  adapter_save_path: "adapters/"
```

---

## Training

Spectra learns from your conversations using LoRA (Low-Rank Adaptation) fine-tuning:

1. **Auto-trigger** — After every 50 conversation turns (configurable), Spectra
   asks whether you want to fine-tune.
2. **Manual trigger** — Type `/train` at any time.
3. **Command-line** — Pass `--train-now` to train immediately on startup.

Training uses QLoRA (8-bit base model + LoRA adapters) and runs entirely on your
GPU. A typical session with 100 turns takes 5-15 minutes on a GTX 1660 Super.

Adapters are saved to the `adapters/` directory and loaded automatically on next
startup.

---

## Troubleshooting

### "CUDA out of memory" during chat
- The model automatically retries with half the token budget.
- If it keeps occurring, reduce `max_new_tokens` in `config.yaml`.

### "CUDA out of memory" during training
- Reduce `epochs` or increase `gradient_accumulation_steps` in the trainer.
- Training is the most VRAM-intensive operation — close other GPU apps first.

### Spotify track not detected
- Ensure Spotify is running as a foreground app.
- On non-Windows systems, only presence detection is supported (no track info).

### Model download fails
- Check your internet connection for the first run.
- The model is cached in `~/.cache/huggingface/` after the first download.

### `bitsandbytes` installation errors
- Make sure your CUDA toolkit version matches your PyTorch build.
- See: https://github.com/TimDettmers/bitsandbytes

---

## Roadmap

- [ ] GUI overlay (system tray icon)
- [ ] Voice input / output (Whisper + TTS)
- [ ] Calendar / task integration
- [ ] Multi-model support (Llama, Mistral, Phi)
- [ ] Android companion app

---

## License

MIT — free to use, modify, and distribute.
