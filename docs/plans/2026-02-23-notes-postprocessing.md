# Notes Post-Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional `generate_notes=true` flag to all three LLM transcription endpoints that post-processes the transcript through a local gpt-oss-20b model and returns + saves structured markdown notes.

**Architecture:** A new `llama-cpp` Docker Compose service runs `ghcr.io/ggml-org/llama.cpp:server-cuda` with the gpt-oss-20b MXFP4 GGUF on GPU 0 (24GB card). The existing `yt-llm-service` calls it via HTTP (async) after transcription completes. A new `NotesService` class handles prompt construction, async HTTP communication, and graceful degradation. The `yt-llm-service` declares `depends_on` with a health condition so it waits for llama-cpp to be ready.

**Tech Stack:** Python 3.11, FastAPI, `httpx.AsyncClient` (already in `requirements.txt`), `ghcr.io/ggml-org/llama.cpp:server-cuda`, gpt-oss-20b MXFP4 GGUF (~12GB), llama-server `--jinja` flag for Harmony chat template.

**Codex review fixes applied:**
1. `.gitkeep` / `.gitignore` moved to Task 0 (before any commit references it)
2. llama.cpp image validated for `huggingface-cli` and `curl` availability; Dockerfile added as fallback
3. `NotesService.generate()` changed to async (`httpx.AsyncClient`) to avoid blocking FastAPI event loop
4. Redundant named `volumes:` block removed from `docker-compose.yml`
5. `generate_notes` wired into all three LLM endpoints (including `/transcribe-file-llm`)
6. `NOTES_MAX_TOKENS` default lowered to `80000` to leave headroom for system prompt + output

---

## Pre-Flight Checklist

Before starting, verify these are true in the working environment:

```bash
# Confirm you're in the project root
ls src/run_llm_api.py   # should exist

# Confirm GPU setup
nvidia-smi              # should show two GPUs (24GB + 8GB)

# Confirm Docker is running
docker ps

# Confirm you have a HuggingFace token (needed for model download)
echo $HF_TOKEN          # should not be empty

# Confirm the llama.cpp server-cuda image has the tools we need
docker run --rm --entrypoint="" ghcr.io/ggml-org/llama.cpp:server-cuda \
  sh -c "which huggingface-cli && which curl && echo ALL_OK" 2>/dev/null \
  || echo "MISSING_TOOLS - see Task 1 Step 1b for fallback"
```

If the last command prints `MISSING_TOOLS`, follow the Dockerfile fallback in Task 1 Step 1b.

---

## Task 0: Create models directory and update .gitignore

**Why first:** Task 1's commit references `models/.gitkeep`. It must exist before that commit.

**Files:**
- Create: `models/.gitkeep`
- Modify: `.gitignore`

**Step 1: Create the models directory placeholder**

```bash
mkdir -p models
touch models/.gitkeep
```

**Step 2: Add model file patterns to `.gitignore`**

Open `.gitignore` and append at the end:

```
# Downloaded ML models (large binary files, never commit)
models/*.gguf
models/*.bin
models/*.safetensors
models/*.ggml
```

**Step 3: Commit**

```bash
git add models/.gitkeep .gitignore
git commit -m "chore: add models directory placeholder, ignore GGUF/binary model files"
```

---

## Task 1: Add llama-cpp Docker service

**Files:**
- Create: `llama-cpp/entrypoint.sh`
- Modify: `docker-compose.yml`

### Step 1a: Validate the base image has required tools

Run the pre-flight check from above. If it prints `ALL_OK`, proceed to Step 2.

### Step 1b: If tools are missing -- create a minimal wrapper Dockerfile

If `huggingface-cli` or `curl` is absent from the base image, create `llama-cpp/Dockerfile`:

```dockerfile
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install missing tools: huggingface-cli (for model download) + curl (for healthcheck)
USER root
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl python3-pip && \
    pip3 install --no-cache-dir "huggingface_hub[cli]" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

If you created this Dockerfile, use `build: ./llama-cpp` instead of `image:` in docker-compose.yml (see Step 2 note).

### Step 2: Create the entrypoint script

```bash
mkdir -p llama-cpp
```

Create `llama-cpp/entrypoint.sh` with this exact content:

```bash
#!/bin/bash
set -e

MODEL_DIR="/models"
MODEL_FILE="openai_gpt-oss-20b-MXFP4.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
REPO="bartowski/openai_gpt-oss-20b-GGUF"

echo "=== llama-cpp entrypoint ==="
echo "Checking for model at: ${MODEL_PATH}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Model not found. Downloading ${MODEL_FILE} from HuggingFace (~12GB)..."
    echo "This may take several minutes on first run."
    huggingface-cli download "${REPO}" \
        --include "${MODEL_FILE}" \
        --local-dir "${MODEL_DIR}"
    echo "Download complete."
else
    echo "Model found. Skipping download."
fi

GPU_LAYERS="${LLAMA_CPP_GPU_LAYERS:-99}"
echo "Starting llama-server with ${GPU_LAYERS} GPU layers..."

exec llama-server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 131072 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --n-gpu-layers "${GPU_LAYERS}" \
    --jinja \
    --log-disable
```

Make it executable:

```bash
chmod +x llama-cpp/entrypoint.sh
```

### Step 3: Add the llama-cpp service to docker-compose.yml

Open `docker-compose.yml`. Currently it has one service (`yt-llm-service`).

**3a.** Add `depends_on` to the existing `yt-llm-service` block, right after `restart: unless-stopped`:

```yaml
    depends_on:
      llama-cpp:
        condition: service_healthy
```

**3b.** Add the new `llama-cpp` service block after the `yt-llm-service` block.

If pre-flight check passed (tools present in base image):
```yaml
  llama-cpp:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    container_name: llama-cpp
    restart: unless-stopped
    entrypoint: ["/bin/bash", "/entrypoint.sh"]
    ports:
      - "8080:8080"
    volumes:
      - ./llama-cpp/entrypoint.sh:/entrypoint.sh:ro
      - ./models:/models
    environment:
      - LLAMA_CPP_GPU_LAYERS=${LLAMA_CPP_GPU_LAYERS:-99}
      - HF_TOKEN=${HF_TOKEN:-}
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      start_period: 300s
      retries: 10
```

If pre-flight check failed (you created `llama-cpp/Dockerfile`), replace `image:` with `build:`:
```yaml
  llama-cpp:
    build: ./llama-cpp
    container_name: llama-cpp
    # ... rest is identical
```

**Important:** Do NOT add a top-level `volumes:` section. The `./models:/models` bind-mount is sufficient; a named volume is not needed here.

### Step 4: Verify the compose file is valid

```bash
docker-compose config --quiet && echo "Valid" || echo "INVALID - fix syntax errors"
```

Expected output: `Valid`

### Step 5: Commit

```bash
git add llama-cpp/ docker-compose.yml
git commit -m "feat: add llama-cpp sidecar service with gpt-oss-20b GGUF download"
```

---

## Task 2: Add configuration fields for notes service

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`

**Note on `LLAMA_CPP_GPU_LAYERS` in config:** This field is stored in `Config` for completeness and logging, but it is only passed to the llama-cpp container via environment variable in `docker-compose.yml`. The Python app itself never uses its value at runtime.

### Step 1: Add two new fields to `Config.__init__` in `src/config.py`

After the `LLM_MERGE_CONSECUTIVE_SPEAKERS` line (line 34), add:

```python
        # Notes service configuration
        self.LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://llama-cpp:8080")
        self.NOTES_MAX_TOKENS = int(os.getenv("NOTES_MAX_TOKENS", "80000"))
        # LLAMA_CPP_GPU_LAYERS is consumed by docker-compose, stored here for logging only
        self.LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_CPP_GPU_LAYERS", "99"))
```

### Step 2: Update the `__str__` method in `src/config.py`

Change the return line to:

```python
        return f"Config(DEVICE={self.DEVICE}, WHISPER_MODEL={self.WHISPER_MODEL}, BATCH_SIZE={self.BATCH_SIZE}, LLM_OUTPUT_FORMAT={self.LLM_OUTPUT_FORMAT}, LLAMA_CPP_URL={self.LLAMA_CPP_URL})"
```

### Step 3: Add the new variables to `.env.example`

At the end of `.env.example`, append:

```
# Notes Service Configuration (llama-cpp sidecar)
LLAMA_CPP_URL=http://llama-cpp:8080
# Base URL for the llama.cpp OpenAI-compatible API (Docker internal network)
# Change to http://localhost:8080 if running llama-cpp outside Docker

LLAMA_CPP_GPU_LAYERS=99
# Layers to offload to GPU (99 = all). Reduce if VRAM is tight (e.g., 32).
# This is also passed directly to the llama-cpp container via docker-compose.

NOTES_MAX_TOKENS=80000
# Max transcript tokens before truncation. Default 80k leaves headroom for
# system prompt (~500 tokens) + generated notes (~4096 tokens) within 128k ctx.
# First/last 25% of text is preserved when truncating.
```

### Step 4: Commit

```bash
git add src/config.py .env.example
git commit -m "feat: add notes service config fields (LLAMA_CPP_URL, NOTES_MAX_TOKENS)"
```

---

## Task 3: Implement NotesService (async)

**Files:**
- Create: `src/notes_service.py`

**Critical:** The `generate()` method must be `async` because FastAPI endpoints are `async def`. Using a synchronous `httpx.Client` inside an async handler blocks the entire event loop for the duration of the LLM call (30-300 seconds), preventing the server from handling any other requests. Use `httpx.AsyncClient` instead.

### Step 1: Create `src/notes_service.py`

```python
"""
Notes generation service using local gpt-oss-20b via llama.cpp HTTP API.

The llama-cpp sidecar exposes an OpenAI-compatible /v1/chat/completions endpoint.
gpt-oss models require the Harmony chat template, which is handled automatically
by llama-server's --jinja flag (the template is embedded in the GGUF).

IMPORTANT: generate() is async to avoid blocking FastAPI's event loop during
the long HTTP call to llama-cpp (model generation can take 30-300 seconds).
"""

import logging
from typing import Optional

import httpx

from config import Config
from simple_logger import log_action

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a note-taking assistant. Given a video transcript, produce a structured "
    "markdown document with: a title inferred from the content, an Overview section "
    "with a 2-3 sentence summary, and topical sections each containing a brief "
    "narrative paragraph and one relevant verbatim quote from the transcript. "
    "Output only the markdown document. No preamble, no explanation."
)

TRUNCATION_NOTICE = (
    "\n\n[NOTE: Transcript was truncated due to length. "
    "The middle portion has been omitted. Analysis covers the beginning and end.]\n\n"
)


class NotesService:
    """Generates structured notes from a transcript using gpt-oss-20b via llama.cpp."""

    def __init__(self, config: Config):
        self.base_url = config.LLAMA_CPP_URL.rstrip("/")
        self.max_tokens = config.NOTES_MAX_TOKENS
        # connect=30s: llama-server should be up (Docker healthcheck enforces this)
        # read=300s: generation on a 20B model can be slow for long transcripts
        self.timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    def _truncate_transcript(self, text: str) -> str:
        """
        If transcript exceeds max_tokens (approximated as words * 1.3),
        preserve the first 25% and last 25%, truncating the middle.
        """
        approx_tokens = len(text.split()) * 1.3
        if approx_tokens <= self.max_tokens:
            return text

        logger.warning(
            f"Transcript too long (~{int(approx_tokens)} tokens). Truncating middle."
        )
        words = text.split()
        keep = int(len(words) * 0.25)
        first_part = " ".join(words[:keep])
        last_part = " ".join(words[-keep:])
        return first_part + TRUNCATION_NOTICE + last_part

    async def generate(self, transcript_text: str) -> Optional[str]:
        """
        Generate structured markdown notes from a transcript.

        This is async: it must be awaited from FastAPI async endpoints.
        Using AsyncClient avoids blocking the event loop during LLM generation.

        Args:
            transcript_text: Plain text transcript content.

        Returns:
            Markdown string with structured notes, or None if generation fails.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("Empty transcript passed to NotesService.generate()")
            return None

        log_action("Generating structured notes from transcript")

        transcript = self._truncate_transcript(transcript_text)

        payload = {
            # llama-server ignores the model field but it's required by the spec
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        url = f"{self.base_url}/v1/chat/completions"
        logger.info(f"Calling llama-cpp at {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error("llama-cpp returned empty choices list")
                return None

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.error("llama-cpp returned empty content in choice[0]")
                return None

            logger.info(f"Notes generated successfully ({len(content)} chars)")
            return content

        except httpx.ConnectError as e:
            logger.warning(f"llama-cpp service unavailable: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.warning(f"llama-cpp request timed out: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                f"llama-cpp HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling llama-cpp: {e}")
            return None
```

### Step 2: Verify the file parses cleanly

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from notes_service import NotesService; print('OK')"
```

Expected output: `OK`

### Step 3: Commit

```bash
git add src/notes_service.py
git commit -m "feat: implement async NotesService for gpt-oss-20b transcript post-processing"
```

---

## Task 4: Add `save_notes()` to StorageService

**Files:**
- Modify: `src/storage_service.py`

### Step 1: Add `save_notes` method to the `StorageService` class

After the `list_transcriptions` method (the last method, around line 260), add:

```python
    def save_notes(self, media_filename: str, notes_text: str) -> str:
        """
        Save generated notes to disk alongside the transcription.

        Always writes to notes.md (overwrites if re-generated). Unlike
        transcription files which are versioned (transcription_1.md etc.),
        notes represent the latest generation and overwriting is intentional.

        Args:
            media_filename: Original media filename (same key as save_transcription)
            notes_text: Markdown notes content from NotesService

        Returns:
            Absolute path to saved notes file

        Raises:
            OSError: If file operations fail
        """
        try:
            directory = self.get_directory_path(media_filename)
            directory.mkdir(parents=True, exist_ok=True)

            file_path = directory / "notes.md"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = f"<!-- Generated: {timestamp} -->\n\n{notes_text}\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Notes saved to: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save notes for {media_filename}: {str(e)}")
            raise OSError(f"Failed to save notes: {str(e)}")
```

### Step 2: Verify the file parses cleanly

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from storage_service import StorageService; print('OK')"
```

Expected output: `OK`

### Step 3: Commit

```bash
git add src/storage_service.py
git commit -m "feat: add save_notes() method to StorageService"
```

---

## Task 5: Wire NotesService into all three LLM FastAPI endpoints

**Files:**
- Modify: `src/run_llm_api.py`

All three LLM endpoints (`/transcribe-llm`, `/transcribe-youtube-llm`, `/transcribe-file-llm`) must get the `generate_notes` flag for API consistency. There are 8 precise changes.

### Step 1: Add the `NotesService` import

After the existing imports (around line 17), add:

```python
from notes_service import NotesService
```

### Step 2: Initialize `NotesService` (around line 31, after `audio_downloader`)

```python
notes_service = NotesService(config)
```

### Step 3: Add `generate_notes` field to `LLMTranscriptionRequest` (around line 92)

Add one line at the end of the existing class body:

```python
    generate_notes: bool = False
```

The full class should look like:

```python
class LLMTranscriptionRequest(BaseModel):
    """Request model for LLM-optimized transcription"""

    audio_file_path: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: str = "simple"
    remove_filler_words: bool = False
    merge_consecutive_speakers: bool = True
    verbose: bool = True
    generate_notes: bool = False
```

### Step 4: Add `generate_notes` field to `YouTubeLLMTranscriptionRequest` (around line 98)

Same change -- add `generate_notes: bool = False` at the end of the class body.

### Step 5: Add `notes` field to `LLMTranscriptionResponse` (around line 111)

Add one line at the end of the existing class body:

```python
    notes: Optional[str] = None
```

### Step 6: Wire notes into `/transcribe-llm` endpoint

This endpoint uses `async def` (line ~193). Find the save-to-disk `try:` block (around line 264). After the existing transcription save block and before `return LLMTranscriptionResponse(...)`, add:

```python
        # Generate and save notes if requested (non-fatal: notes failure doesn't break transcription)
        notes_text = None
        if request.generate_notes:
            try:
                transcript_text = llm_result.get("text") or ""
                if not transcript_text and "blocks" in llm_result:
                    transcript_text = " ".join(
                        b.get("text", "") for b in llm_result.get("blocks", [])
                    )
                notes_text = await notes_service.generate(transcript_text)
                if notes_text and saved_path:
                    notes_path = transcription_service.storage_service.save_notes(
                        media_filename=request.audio_file_path,
                        notes_text=notes_text,
                    )
                    logger.info(f"Notes saved to: {notes_path}")
            except Exception as e:
                logger.warning(f"Notes generation failed (non-fatal): {e}")

        response_data["notes"] = notes_text
```

### Step 7: Wire notes into `/transcribe-youtube-llm` endpoint

Same pattern. In the `transcribe_youtube_llm` function (around line 366), after the save-to-disk block (around line 441), add the identical block but use `storage_name` (already set to `video_title` in that function):

```python
        # Generate and save notes if requested
        notes_text = None
        if request.generate_notes:
            try:
                transcript_text = llm_result.get("text") or ""
                if not transcript_text and "blocks" in llm_result:
                    transcript_text = " ".join(
                        b.get("text", "") for b in llm_result.get("blocks", [])
                    )
                notes_text = await notes_service.generate(transcript_text)
                if notes_text and saved_path:
                    notes_path = transcription_service.storage_service.save_notes(
                        media_filename=storage_name,
                        notes_text=notes_text,
                    )
                    logger.info(f"Notes saved to: {notes_path}")
            except Exception as e:
                logger.warning(f"Notes generation failed (non-fatal): {e}")

        response_data["notes"] = notes_text
```

### Step 8: Wire notes into `/transcribe-file-llm` endpoint

The `transcribe_file_llm` function (around line 547) currently accepts form parameters, not a Pydantic model. Add `generate_notes` as a new Form parameter alongside the others:

In the function signature, add:
```python
    generate_notes: bool = Form(False),
```

Then after the existing save-to-disk block (around line 649), add the same notes block, using `file.filename` as the storage key:

```python
        # Generate and save notes if requested
        notes_text = None
        if generate_notes:
            try:
                transcript_text = llm_result.get("text") or ""
                if not transcript_text and "blocks" in llm_result:
                    transcript_text = " ".join(
                        b.get("text", "") for b in llm_result.get("blocks", [])
                    )
                notes_text = await notes_service.generate(transcript_text)
                if notes_text and saved_path:
                    notes_path = transcription_service.storage_service.save_notes(
                        media_filename=file.filename,
                        notes_text=notes_text,
                    )
                    logger.info(f"Notes saved to: {notes_path}")
            except Exception as e:
                logger.warning(f"Notes generation failed (non-fatal): {e}")

        response_data["notes"] = notes_text
```

### Step 9: Update the root endpoint description

In the `/` root endpoint (around line 791), update `llm_endpoints`:

```python
        "llm_endpoints": {
            "description": "Endpoints that format transcription data for LLM consumption",
            "formats": ["simple", "speaker", "structured", "markdown"],
            "features": [
                "filler word removal",
                "speaker merging",
                "clean text output",
                "structured notes via generate_notes=true (requires llama-cpp sidecar)",
            ]
        }
```

### Step 10: Verify the file parses cleanly

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); import ast; ast.parse(open('src/run_llm_api.py').read()); print('Syntax OK')"
```

Expected output: `Syntax OK`

### Step 11: Commit

```bash
git add src/run_llm_api.py
git commit -m "feat: wire generate_notes flag into all three LLM transcription endpoints"
```

---

## Task 6: Smoke test the full stack

### Step 1: Build and start services

```bash
docker-compose up --build -d
```

The `yt-llm-service` will wait for `llama-cpp` to pass its healthcheck before starting (due to `depends_on` with `service_healthy`).

Watch logs:

```bash
# Terminal 1: watch llama-cpp (first run downloads 12GB model, takes 5-15 min)
docker-compose logs -f llama-cpp

# Terminal 2: watch transcription service
docker-compose logs -f yt-llm-service
```

Wait until llama-cpp logs: `llama server listening at http://0.0.0.0:8080`

### Step 2: Verify both services are healthy

```bash
curl -s http://localhost:8002/health | python3 -m json.tool
curl -s http://localhost:8080/health | python3 -m json.tool
```

Expected: `{"status": "ok"}` from llama-cpp; `{"status": "ok", "service": "yt-llm-transcription", ...}` from yt-llm-service.

### Step 3: Test with a local file (deterministic -- no network dependency)

There is already an uploaded file at `data/uploads/1475.mp4`. Use it:

```bash
curl -s -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@data/uploads/1475.mp4" \
  -F "output_format=simple" \
  -F "generate_notes=true" \
  | python3 -m json.tool
```

Expected response structure:

```json
{
  "success": true,
  "text": "...",
  "language": "...",
  "metadata": {...},
  "notes": "# Title\n\n## Overview\n...",
  "error": null
}
```

Verify the notes file on disk:

```bash
ls data/output/
# Should show a directory named after the sanitized filename
find data/output/ -name "notes.md" | xargs head -5
```

### Step 4: Test graceful degradation (llama-cpp down)

```bash
docker-compose stop llama-cpp

curl -s -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@data/uploads/1475.mp4" \
  -F "output_format=simple" \
  -F "generate_notes=true" \
  | python3 -m json.tool
```

Expected: `"success": true`, `"notes": null` -- transcription completes normally.

```bash
docker-compose start llama-cpp
```

### Step 5: Test YouTube URL (optional -- requires network and YouTube auth)

```bash
curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple",
    "generate_notes": true
  }' | python3 -m json.tool
```

### Step 6: Verify zero overhead when `generate_notes` is not set

```bash
curl -s -X POST http://localhost:8002/transcribe-file-llm \
  -F "file=@data/uploads/1475.mp4" \
  -F "output_format=simple" \
  | python3 -m json.tool

# Check logs: no llama-cpp calls should appear
docker-compose logs yt-llm-service | grep -i "llama\|notes" | tail -10
```

Expected: no llama-cpp log lines for this request.

### Step 7: Final commit

```bash
git add .
git commit -m "feat: notes post-processing via gpt-oss-20b - complete implementation"
```

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `llama-cpp` exits immediately | `entrypoint.sh` has Windows line endings | `dos2unix llama-cpp/entrypoint.sh` |
| `huggingface-cli: not found` | Base image missing CLI | Follow Task 1 Step 1b (create Dockerfile) |
| `curl: not found` (healthcheck fails) | Base image missing curl | Follow Task 1 Step 1b (create Dockerfile) |
| Model download fails | Missing `HF_TOKEN` | Set `HF_TOKEN` in `.env` |
| `yt-llm-service` won't start | `llama-cpp` healthcheck not passing | Wait or check llama-cpp logs |
| `notes: null` despite llama-cpp running | Cold start: model still loading | Wait for `llama server listening` in logs |
| `RuntimeError: Event loop is closed` | Using sync httpx inside async | Confirm `generate()` uses `httpx.AsyncClient` with `await` |
| CUDA OOM during notes | WhisperX still holding VRAM | Check if model is cached; transcription should release GPU first |
| Notes output is garbled / not markdown | `--jinja` missing from llama-server | Confirm entrypoint.sh has `--jinja` flag |
| `422 Unprocessable Entity` on file upload with `generate_notes` | `generate_notes` not added to form params | Confirm `generate_notes: bool = Form(False)` is in the function signature |
