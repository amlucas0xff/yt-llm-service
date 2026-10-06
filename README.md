# YT-LLM Transcription Service

A high-performance FastAPI service for YouTube audio transcription with advanced speaker diarization, LLM-optimized output formatting, and automated structured notes generation via a local LLM sidecar.

> **IMPORTANT: NVIDIA GPU required**
> Both services share one GPU by default: WhisperX and the gpt-oss-20b notes sidecar together want ~20GB of VRAM, so a 24GB card is the comfortable target. Splitting them across two GPUs is supported — see [GPU Setup](#gpu-setup). CPU fallback works but is not recommended for production use.

## Features

### Core Capabilities
- **YouTube Audio Processing**: Direct download and transcription from YouTube URLs
- **File Upload Support**: Transcribe uploaded video/audio files
- **Speaker Diarization**: Advanced speaker separation using pyannote.audio
- **Multiple Output Formats**: Simple text, speaker-separated, structured JSON, and Markdown
- **LLM-Optimized Outputs**: Clean, formatted transcriptions ready for AI processing
- **GPU Acceleration**: CUDA support for high-speed processing
- **Persistent Storage**: Automatic saving and organization of transcriptions

### Advanced Features
- **Structured Notes Generation**: Optional post-processing via local gpt-oss-20b (llama-cpp sidecar) — produces a markdown document with title, overview, and topical sections
- **Filler Word Removal**: Intelligent removal of "um", "uh", and other speech disfluencies
- **Speaker Merging**: Automatic merging of consecutive segments from the same speaker
- **Batch Processing**: Configurable batch sizes for optimal performance
- **VRAM Idle Unload**: llama-cpp automatically unloads the model after idle timeout, reclaiming GPU memory on shared hardware
- **RESTful API**: Complete FastAPI implementation with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment with GPU passthrough

## Architecture

Two Docker services, both on GPU 0:

| Service | Host port | GPU | Model | Role |
|---------|-----------|-----|-------|------|
| `yt-llm-service` | 8002 | GPU 0 | WhisperX large-v3-turbo | Transcription + speaker diarization |
| `llama-cpp` | 18080 | GPU 0 | gpt-oss-20b MXFP4 | Structured notes generation |

`yt-llm-service` depends on `llama-cpp` (Docker healthcheck enforced). Notes generation is non-fatal — if llama-cpp is unavailable, transcription still succeeds and `notes` returns `null`.

The two models coexist on one 24GB card because llama-cpp unloads gpt-oss-20b from VRAM after `LLAMA_CPP_IDLE_SECONDS`. `llama-cpp`'s host port is only for hitting the sidecar by hand — `yt-llm-service` talks to it over the compose network at `http://llama-cpp:8080`.

### YouTube URL Pipeline

```
YouTube URL
    │
    ├─── yt-dlp metadata fetch (title, channel, tags, description)
    │                                          │
    └─── yt-dlp audio download                 │
                    │                          │
          WhisperX transcription               │
                    │                          │
                    ├──────────────────────────┘
                    │        (metadata grounds proper-noun spelling)
    ┌───────────────┴───────────────┐
    │                               │
disk storage              gpt-oss-20b notes
(transcription.txt)      (structured markdown)
```

File upload endpoints (`/transcribe-file-llm`) follow the same path minus the metadata fetch.

## Tech Stack

- **Framework**: FastAPI with async/await support
- **Transcription**: WhisperX (large-v3-turbo), pyannote.audio for diarization
- **Notes LLM**: llama.cpp server (gpt-oss-20b MXFP4 GGUF, OpenAI-compatible API)
- **Audio Processing**: yt-dlp for YouTube downloads, ffmpeg for format conversion
- **GPU Support**: CUDA/cuDNN with PyTorch backend
- **Containerization**: Docker Compose with NVIDIA runtime support
- **Language**: Python 3.11+ with type hints throughout

## 🚀 Quick Start

### How It Works

![Service Workflow](docs/images/workflow-diagram.svg)

### Prerequisites

- Docker and docker-compose
- **NVIDIA GPU with CUDA support (REQUIRED for optimal performance)**
  - RTX 3000 series or newer recommended
  - Minimum 8GB VRAM for large-v3-turbo model
  - NVIDIA Container Toolkit for Docker GPU access
- HuggingFace account (free) - get token from https://huggingface.co/settings/tokens
- CPU-only mode available but **NOT recommended** (10-20x slower)

### Installation (Docker - Recommended)

**Why Docker?** This project has complex dependencies (WhisperX, PyTorch, CUDA, ffmpeg, yt-dlp). Docker handles everything automatically.

1. **Clone the repository**
```bash
git clone https://github.com/amlucas0xff/yt-llm-service.git
cd yt-llm-service
```

2. **Configure environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your HuggingFace token
nano .env  # or vim, code, etc.
# Set: HF_TOKEN=your_token_here
```

3. **Download the notes model** (~12GB, once per machine)
```bash
curl -L -o models/openai_gpt-oss-20b-MXFP4.gguf \
  "https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-MXFP4.gguf"
```

`models/` is mounted into the sidecar, so the file survives image rebuilds. No HuggingFace token is needed for this one. This step is not optional: `llama-cpp` exits with instructions if the file is missing, and `yt-llm-service` waits on its healthcheck, so the whole stack stays down until the model is in place.

4. **Start the service**
```bash
docker-compose up --build
```

The service will be available at `http://localhost:8002`

**First run:** Docker will download the WhisperX models (~2-3GB). This may take several minutes.

### Quick Test

```bash
# Check service health
curl http://localhost:8002/health

# Transcribe a YouTube video
curl -X POST "http://localhost:8002/transcribe-youtube-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple"
  }'

# Transcribe + structured notes
curl -X POST "http://localhost:8002/transcribe-youtube-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple",
    "generate_notes": true
  }'
```

## API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### Core Endpoints

#### 1. YouTube Transcription (LLM-Optimized)
```bash
POST /transcribe-youtube-llm
```

**Request fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `youtube_url` | string | required | YouTube URL |
| `output_format` | string | `simple` | `simple`, `speaker`, `structured`, `markdown` |
| `generate_notes` | bool | `false` | Generate structured notes via llama-cpp |
| `remove_filler_words` | bool | `false` | Strip "um", "uh", etc. |
| `merge_consecutive_speakers` | bool | `true` | Merge adjacent segments from same speaker |
| `min_speakers` / `max_speakers` | int | `null` | Speaker count hints for diarization |

**Example response with notes:**
```json
{
  "success": true,
  "text": "So Anthropic has released a list of really interesting updates...",
  "language": "en",
  "notes": "# Anthropic Tool Calling Updates\n\n## Overview\n...",
  "video_metadata": {
    "title": "Anthropic Tool Calling Updates",
    "channel": "AI Jason",
    "tags": ["anthropic", "claude"],
    "categories": ["Science & Technology"],
    "description": "..."
  },
  "metadata": {
    "video_id": "3wglqgskzjQ",
    "duration": 847,
    "speakers_detected": 1,
    "word_count": 2341
  }
}
```

`video_metadata` is passed to the notes prompt as authoritative ground truth, so proper nouns in the notes follow the video's own spelling even when WhisperX mishears them.

Notes are also saved to `data/output/<title>/notes.md` alongside the transcription files.

#### 2. File Upload Transcription
```bash
POST /transcribe-file-llm
```

Multipart form upload. Accepts the same parameters as above (`generate_notes`, `output_format`, etc.).

```bash
curl -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@video.mp4" \
  -F "output_format=simple" \
  -F "generate_notes=true"
```

#### 3. Audio File Path Transcription
```bash
POST /transcribe-llm
```

Transcribe a file already on the server by path. Same request shape as YouTube but with `audio_file_path` instead of `youtube_url`.

#### 4. Health Check
```bash
GET /health
```

Returns service status and GPU information.

### Output Formats

- **simple**: Clean text without speaker labels
- **speaker**: Text with speaker identification
- **structured**: JSON with detailed segment information
- **markdown**: Formatted Markdown with speaker headers

## Configuration

### Environment Variables

**yt-llm-service:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Processing device (`cuda`/`cpu`) |
| `COMPUTE_TYPE` | `float16` | Compute precision |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model size |
| `BATCH_SIZE` | `16` | Processing batch size |
| `HF_TOKEN` | - | HuggingFace API token (required for diarization) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LLAMA_CPP_URL` | `http://llama-cpp:8080` | URL of the llama-cpp sidecar |
| `NOTES_MAX_TOKENS` | `80000` | Max transcript tokens before middle truncation |

**Host paths** (read by `docker-compose.yml`, not by the service):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_OUTPUT_DIR` | `./data/output` | Host directory mounted as `/app/output`. **Set an absolute path** — provenance links in vault notes are omitted when this is relative, since a relative path resolves only inside the repo |
| `HOST_OBSIDIAN_VAULT` | `/tmp/no-vault` | Absolute path to your Obsidian vault. Must equal `vault_path` in `config.toml`; the export silently does nothing if unset |
| `HOST_YT_LLM_CONFIG` | `~/.config/yt-llm` | Directory holding `config.toml` |

**llama-cpp sidecar:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_GPU_LAYERS` | `99` | GPU layers to offload (99 = all) |
| `LLAMA_CPP_IDLE_SECONDS` | `5` | Seconds idle before VRAM unload (-1 only with split GPUs) |
| `SIDECAR_GPU_SEPARATE` | `false` | API skips GPU wait only with an explicitly pinned split-GPU override |
| `LLAMA_CPP_HOST_PORT` | `18080` | Host port for reaching the sidecar directly |

See `.env.example` for complete configuration options.

### Performance Tuning

#### GPU Memory Optimization (Recommended)
```bash
# For 8GB GPU (RTX 3060, RTX 3070)
DEVICE=cuda
BATCH_SIZE=8
COMPUTE_TYPE=float16

# For 12-16GB GPU (RTX 3080, RTX 3090)
DEVICE=cuda
BATCH_SIZE=16
COMPUTE_TYPE=float16

# For 24GB+ GPU (RTX 4090, A5000)
DEVICE=cuda
BATCH_SIZE=32
COMPUTE_TYPE=float16
```

#### CPU Fallback (Not Recommended for Production)
```bash
# CPU-only mode (10-20x slower, use only if no GPU available)
DEVICE=cpu
COMPUTE_TYPE=float32
BATCH_SIZE=4
```

## Obsidian Integration

Generated notes can be automatically mirrored to your Obsidian vault.

1. Copy the example config:
   ```bash
   mkdir -p ~/.config/yt-llm
   cp config.example.toml ~/.config/yt-llm/config.toml
   ```
2. Edit `~/.config/yt-llm/config.toml`. Use an **absolute** `vault_path` — it is resolved inside the container, and `~` there is not your home directory:
   ```toml
   [obsidian]
   enabled = true
   vault_path = "/home/you/Documents/obsidian"   # absolute path to your vault
   inbox_dir = "Inbox"                            # subdirectory inside vault
   tags = ["video-notes"]
   ```
3. Point the container at the same vault, in `.env`:
   ```bash
   HOST_OBSIDIAN_VAULT=/home/you/Documents/obsidian   # must equal vault_path above
   ```

   This is the step that makes the export work at all. The vault is bind-mounted at the *same absolute path* it has on the host, so `vault_path` is valid on both sides. Leave `HOST_OBSIDIAN_VAULT` unset and the mount falls back to `/tmp/no-vault`, `vault_path` does not exist in the container, and `save_note()` silently does nothing. Set `HOST_YT_LLM_CONFIG` too if `config.toml` lives outside `~/.config/yt-llm`.

   For provenance links to be followable, `HOST_OUTPUT_DIR` must also be absolute — see [Environment Variables](#environment-variables).

4. Restart the stack (`docker compose up -d`) so the new mounts take effect, then run `transcribe` as usual. Notes are saved to `{vault_path}/{inbox_dir}/{Title}.md` with YAML frontmatter:

   ```yaml
   ---
   date: 2026-08-11
   source: https://youtu.be/abc123
   source_transcript: '/home/you/Documents/obsidian/transcripts/Understanding Transformers/transcription_1.md'
   truncated: false
   tags:
     - video-notes
   ---
   ```

   `source_transcript` points back at the transcript the notes were made from, translated to its **host** path so it opens from the vault, and quoted so a path containing `: ` cannot break the frontmatter. If `HOST_OUTPUT_DIR` is unset or relative, the field is omitted rather than filled with a path that would not resolve. `truncated: true` means the transcript exceeded `NOTES_MAX_TOKENS` and its middle was dropped before the model saw it — the notes then cover only the beginning and end.

On Linux, Compose binds the host's `/etc/localtime` read-only into the API, so new notes use the host-local date. The image has no tzdata, so `TZ` alone is insufficient. The isolated smoke API gets the same mount; existing vault notes are not rewritten.

The integration is silent: if the config file is absent or `enabled = false`, nothing changes.

## Running tests

```bash
make test
```

That is the whole recipe. It runs the unit suite in a throwaway `uv` environment — no virtualenv to activate, no environment variables to set, and none of the ML stack to install (it finishes in seconds).

`conftest.py` does the setup the suite needs: it puts `tests/stubs`, the repo root and `src/` on `sys.path`, points `TEMP_DIR`/`OUTPUT_DIR` at a temporary sandbox, drops a no-op `yt-dlp` shim on `PATH`, and neuters `python-dotenv`. All four matter because `run_llm_api.py` builds a `Config()` and an `AudioDownloader()` at import time.

The last one is what keeps the suite honest: `Config()` calls `load_dotenv()`, which finds the repo-root `.env`, so without it every test inherits your personal `HF_TOKEN` and `HOST_OUTPUT_DIR` and the results stop meaning anything on anyone else's machine. Tests should set the variables they depend on. A test genuinely *about* env loading can ask for the `real_dotenv` fixture.

`tests/stubs/torch.py` shadows the real `torch` so the suite never pulls a CUDA wheel. Every model call is mocked; nothing here exercises WhisperX or llama-cpp for real — that is what the smoke test below is for.

On pushes and pull requests, `.github/workflows/checks.yml` runs the mocked tests, Compose validation and GPU config check with checkout defaults, smoke-script syntax/ShellCheck, and Ruff `E4,E7,E9,F821`. Hosted CI does not run the GPU smoke test, download the 12 GB model, or require `.env` or credentials. Full Ruff `F` is deferred to a separate reviewed cleanup: the wider check currently has 10 baseline findings, including imports used as dependency probes.

Pass pytest arguments through `ARGS`:

```bash
make test ARGS="-k obsidian -v"
```

If you already have the dependencies installed, plain `pytest tests/` works too.

## Smoke test

```bash
make smoke                     # default video
make smoke ARGS="https://youtu.be/..."
```

This is the only check that runs WhisperX and llama-cpp for real. It reuses (or starts) the Compose llama-cpp sidecar and runs a disposable API container on the project's network, bound to a free loopback port. The API uses this checkout's `src/` read-only, plus temporary output, audio, config, and Obsidian vault directories. It never mounts or writes the live output or vault, and it leaves the live API alone. Requests go only to the port assigned to the disposable container. The script removes its container and scratch files on exit, including failure or interruption. It checks HTTP 200, non-empty response transcript and notes, a fresh `notes.md`, and a vault note with the host-local date, valid source and a resolvable transcript link.

The default video is [Me at the zoo](https://www.youtube.com/watch?v=jNQXAC9IVRw) — 19 seconds, has speech, and won't be taken down.

Run it on demand rather than on every commit — it is GPU-bound and holds VRAM for the duration.

**Observed:** 14s warm for the 19s default video (RTX 3090 Ti, both services on one GPU). The first run after `docker compose up` costs ~65s instead, because llama-cpp pages gpt-oss-20b into VRAM and WhisperX loads `large-v3-turbo` on the first request. Longer videos scale with audio length, not with this floor.

| Variable | Default | Description |
|----------|---------|-------------|
| `SMOKE_TIMEOUT` | `900` | Seconds to wait for the transcription response |
| `SMOKE_TEST_FAIL_AFTER_HEALTH` | `0` | Set to `1` to test cleanup after isolated API health without sending a transcription request |

## Project Structure

```
yt-llm-service/
├── src/                         # Core application code
│   ├── run_llm_api.py           # FastAPI service + all endpoints
│   ├── transcription_service.py # WhisperX transcription engine
│   ├── notes_service.py         # Notes generation via llama-cpp
│   ├── audio_downloader.py      # YouTube/file audio extraction
│   ├── storage_service.py       # Result persistence (transcription + notes)
│   ├── obsidian_service.py      # Obsidian vault export
│   └── config.py                # Configuration (env vars)
├── llama-cpp/                   # llama-cpp sidecar service
│   ├── Dockerfile               # Pinned llama.cpp CUDA image (digest-locked)
│   └── entrypoint.sh            # Checks for the model, starts llama-server
├── models/                      # GGUF model files (gitignored, see Quick Start)
├── data/                        # Runtime data (gitignored)
│   ├── output/                  # Saved transcriptions + notes.md
│   ├── tmp/                     # Temporary audio files
│   └── logs/                    # Application logs
├── tests/                       # Unit suite (`make test`)
│   └── stubs/                   # Stand-ins for heavyweight ML packages
├── scripts/smoke.sh             # End-to-end test against the real containers
├── docs/                        # Documentation
├── Makefile                     # `make test`, `make check-config`, `make smoke`
├── conftest.py                  # Test session setup (sys.path, dirs, yt-dlp shim)
├── docker-compose.yml           # Two-service orchestration
├── Dockerfile                   # yt-llm-service container image
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## GPU Setup

`docker-compose.yml` puts both services on GPU 0:
- `yt-llm-service` is explicitly pinned to GPU 0
- `llama-cpp` uses `device_ids: ['0']` (explicitly GPU 0)

Check your layout with:
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

### Splitting across two GPUs

With a second card, moving the notes sidecar off the transcription GPU means neither has to wait on the other's VRAM. Put this in `docker-compose.override.yml` (gitignored, applied automatically by `docker compose`):

```yaml
services:
  llama-cpp:
    deploy:
      resources:
        reservations:
          # !override replaces the list. Without it compose *appends*, and the
          # GPU 0 reservation from docker-compose.yml is still requested.
          devices: !override
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
  yt-llm-service:
    environment:
      SIDECAR_GPU_SEPARATE: "true"
```

The marker only belongs in the API service when the GPUs really are separate. `make check-config` checks both services' resolved GPU reservations; a `count: 1` reservation does not prove isolation. Once split, `LLAMA_CPP_IDLE_SECONDS=-1` is allowed and the API skips the wait. The same override file is the right place for any other machine-local deviation — a different host port or idle timeout.

### Install NVIDIA Container Toolkit

Required for Docker GPU passthrough:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### VRAM Management

The llama-cpp sidecar unloads the model from VRAM after `LLAMA_CPP_IDLE_SECONDS` (default: 5s). Before GPU transcription on a shared GPU, the API polls for up to 30 attempts (~30s) for the sidecar to sleep. The next notes request triggers a cold reload (~15s). On shared hardware, idle must be nonnegative and **less than 30s**; `-1` disables sleep and conflicts with the wait. Before starting transcription, run `make check-config`: it inspects effective Compose settings (including `.env` and overrides) without displaying credentials. It does not start or stop services. If the optional sidecar is unreachable, transcription can still proceed; a reachable but awake sidecar times out rather than risking GPU memory contention. With two explicitly pinned GPUs, use the override above to skip the wait.

Compose passes the same effective idle value to the API, which rejects unsafe shared-GPU values on every CUDA request, including when `/props` reports sleeping. Keep overrides of the idle setting consistent in both services; `make check-config` rejects mismatches. Reachable `/props` errors, malformed state, and read timeouts stop transcription before GPU work; only a connection failure is treated as an optional sidecar being down. The isolated smoke API uses the resolved API GPU reservation instead of `--gpus all`; ambiguous reservations fail closed.

### CPU-Only Fallback (Not Recommended)

Set in `.env`:
```
DEVICE=cpu
COMPUTE_TYPE=float32
BATCH_SIZE=4
```

Processing will be 10-20x slower. Notes generation still requires the llama-cpp sidecar regardless of transcription device.

## 📊 Performance Benchmarks

| Model | GPU | Batch Size | Processing Speed | Accuracy |
|-------|-----|------------|------------------|----------|
| large-v3-turbo | RTX 4090 | 32 | ~10x realtime | Excellent |
| large-v3-turbo | RTX 3080 | 16 | ~6x realtime | Excellent |
| large-v3 | RTX 4090 | 16 | ~4x realtime | Superior |
| CPU-only | Intel i9 | 4 | ~0.5x realtime | Good |

*Benchmarks based on typical YouTube content (10-minute videos)*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for advanced transcription capabilities
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube processing
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

---

**Built with ❤️ for the AI/ML community**