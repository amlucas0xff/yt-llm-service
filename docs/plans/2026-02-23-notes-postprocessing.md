# Notes Post-Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional `generate_notes=true` flag to the LLM transcription endpoints that post-processes the transcript through a local gpt-oss-20b model and returns + saves structured markdown notes.

**Architecture:** A new `llama-cpp` Docker Compose service runs `ghcr.io/ggml-org/llama.cpp:server-cuda` with the gpt-oss-20b MXFP4 GGUF on GPU 0 (24GB card). The existing `yt-llm-service` calls it via HTTP after transcription completes. A new `NotesService` class handles prompt construction, HTTP communication, and graceful degradation.

**Tech Stack:** Python 3.11, FastAPI, `httpx` (async HTTP client already available via `requirements.txt`), `ghcr.io/ggml-org/llama.cpp:server-cuda`, gpt-oss-20b MXFP4 GGUF (~12GB), llama-server `--jinja` flag for Harmony chat template.

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
```

---

## Task 1: Add llama-cpp Docker service

**Files:**
- Create: `llama-cpp/entrypoint.sh`
- Modify: `docker-compose.yml`

**Step 1: Create the llama-cpp directory and entrypoint script**

```bash
mkdir -p llama-cpp models
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

**Step 2: Add the llama-cpp service to docker-compose.yml**

Open `docker-compose.yml`. It currently has one service (`yt-llm-service`). Add the following service block after the existing service, and add a `models` volume at the bottom:

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

Also add this `volumes:` section at the end of `docker-compose.yml` (top-level, same indentation as `services:`):

```yaml
volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./models
```

**Step 3: Verify the compose file is valid**

```bash
docker-compose config --quiet && echo "Valid" || echo "INVALID - fix syntax errors"
```

Expected output: `Valid`

**Step 4: Commit**

```bash
git add llama-cpp/entrypoint.sh docker-compose.yml models/.gitkeep
git commit -m "feat: add llama-cpp sidecar service for notes generation"
```

---

## Task 2: Add configuration fields for notes service

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`

**Step 1: Add three new fields to `Config.__init__` in `src/config.py`**

After the `LLM_MERGE_CONSECUTIVE_SPEAKERS` line (line 34), add:

```python
        # Notes service configuration
        self.LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://llama-cpp:8080")
        self.LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_CPP_GPU_LAYERS", "99"))
        self.NOTES_MAX_TOKENS = int(os.getenv("NOTES_MAX_TOKENS", "100000"))
```

**Step 2: Update the `__str__` method in `src/config.py`**

Change the return line to include the new field:

```python
        return f"Config(DEVICE={self.DEVICE}, WHISPER_MODEL={self.WHISPER_MODEL}, BATCH_SIZE={self.BATCH_SIZE}, LLM_OUTPUT_FORMAT={self.LLM_OUTPUT_FORMAT}, LLAMA_CPP_URL={self.LLAMA_CPP_URL})"
```

**Step 3: Add the new variables to `.env.example`**

At the end of `.env.example`, add:

```
# Notes Service Configuration (llama-cpp sidecar)
LLAMA_CPP_URL=http://llama-cpp:8080
# Base URL for the llama.cpp OpenAI-compatible API (internal Docker network)

LLAMA_CPP_GPU_LAYERS=99
# Number of model layers to offload to GPU (99 = all layers)
# Reduce if you hit VRAM limits (e.g., 32 for partial offload)

NOTES_MAX_TOKENS=100000
# Maximum transcript tokens before truncation (gpt-oss-20b supports 128k context)
# Transcripts over this limit are trimmed: first 25% + last 25% preserved
```

**Step 4: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat: add notes service config fields (LLAMA_CPP_URL, GPU_LAYERS, NOTES_MAX_TOKENS)"
```

---

## Task 3: Implement NotesService

**Files:**
- Create: `src/notes_service.py`

**Step 1: Create `src/notes_service.py`**

```python
"""
Notes generation service using local gpt-oss-20b via llama.cpp HTTP API.

The llama-cpp sidecar exposes an OpenAI-compatible /v1/chat/completions endpoint.
gpt-oss models require the Harmony chat template, which is handled automatically
by llama-server's --jinja flag (embedded in the GGUF).
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
        # httpx timeout: 120s connect, 300s read (model loading + generation)
        self.timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    def _truncate_transcript(self, text: str) -> str:
        """
        If transcript exceeds max_tokens (approximated as words * 1.3),
        preserve the first 25% and last 25%, truncating the middle.
        """
        # Rough approximation: 1 word ≈ 1.3 tokens
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

    def generate(self, transcript_text: str) -> Optional[str]:
        """
        Generate structured markdown notes from a transcript.

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
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
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
            logger.error(f"llama-cpp HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling llama-cpp: {e}")
            return None
```

**Step 2: Verify the file parses cleanly**

```bash
cd /path/to/yt-llm-service
python3 -c "import sys; sys.path.insert(0, 'src'); from notes_service import NotesService; print('OK')"
```

Expected output: `OK`

If you see an ImportError about `httpx`, it is already in `requirements.txt` via `httpx>=0.25.0`. Inside Docker it will be available. For local testing: `pip install httpx`.

**Step 3: Commit**

```bash
git add src/notes_service.py
git commit -m "feat: implement NotesService for gpt-oss-20b transcript post-processing"
```

---

## Task 4: Add `save_notes()` to StorageService

**Files:**
- Modify: `src/storage_service.py`

**Step 1: Add `save_notes` method to the `StorageService` class**

After the `list_transcriptions` method (the last method, around line 260), add:

```python
    def save_notes(self, media_filename: str, notes_text: str) -> str:
        """
        Save generated notes to disk alongside the transcription.

        Args:
            media_filename: Original media filename (same as used in save_transcription)
            notes_text: Markdown notes content

        Returns:
            Path to saved notes file

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

**Step 2: Verify the file parses cleanly**

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from storage_service import StorageService; print('OK')"
```

Expected output: `OK`

**Step 3: Commit**

```bash
git add src/storage_service.py
git commit -m "feat: add save_notes() method to StorageService"
```

---

## Task 5: Wire NotesService into FastAPI endpoints

**Files:**
- Modify: `src/run_llm_api.py`

This is the largest single change. Read the file carefully before editing -- there are 5 precise changes needed.

**Step 1: Add the `NotesService` import**

After the existing service imports (around line 17):

```python
from notes_service import NotesService
```

**Step 2: Initialize `NotesService` after `audio_downloader` initialization (around line 31)**

```python
notes_service = NotesService(config)
```

**Step 3: Add `generate_notes` field to `LLMTranscriptionRequest` (around line 92)**

```python
class LLMTranscriptionRequest(BaseModel):
    """Request model for LLM-optimized transcription"""

    audio_file_path: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: str = "simple"  # simple, speaker, structured, markdown
    remove_filler_words: bool = False
    merge_consecutive_speakers: bool = True
    verbose: bool = True
    generate_notes: bool = False  # ADD THIS LINE
```

**Step 4: Add `generate_notes` field to `YouTubeLLMTranscriptionRequest` (around line 98)**

```python
class YouTubeLLMTranscriptionRequest(BaseModel):
    """Request model for YouTube LLM-optimized transcription"""

    youtube_url: str
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: str = "simple"  # simple, speaker, structured, markdown
    remove_filler_words: bool = False
    merge_consecutive_speakers: bool = True
    verbose: bool = True
    generate_notes: bool = False  # ADD THIS LINE
```

**Step 5: Add `notes` field to `LLMTranscriptionResponse` (around line 111)**

```python
class LLMTranscriptionResponse(BaseModel):
    """Response model for LLM-optimized transcription"""

    success: bool
    text: Optional[str] = None
    speakers: Optional[dict] = None
    blocks: Optional[list] = None
    language: Optional[str] = None
    metadata: dict
    error: Optional[str] = None
    notes: Optional[str] = None  # ADD THIS LINE
```

**Step 6: Wire notes generation into `transcribe_audio_llm` endpoint**

In the `/transcribe-llm` endpoint function (starting around line 193), find the block that saves the transcription to disk (the `try:` block around line 264). After the `if saved_path:` log line and before the `except Exception as e:` for save failure, add the notes generation block:

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
                notes_text = notes_service.generate(transcript_text)
                if notes_text and saved_path:
                    storage_name = request.audio_file_path
                    notes_path = transcription_service.storage_service.save_notes(
                        media_filename=storage_name,
                        notes_text=notes_text
                    )
                    logger.info(f"Notes saved to: {notes_path}")
            except Exception as e:
                logger.warning(f"Notes generation failed (non-fatal): {e}")

        response_data["notes"] = notes_text
```

**Step 7: Wire notes generation into `transcribe_youtube_llm` endpoint**

In the `/transcribe-youtube-llm` endpoint (starting around line 366), find the save-to-disk block (around line 441). After the `if saved_path:` log line, add the identical notes block but using `storage_name` (which is already set to `video_title` in that function):

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
                notes_text = notes_service.generate(transcript_text)
                if notes_text and saved_path:
                    notes_path = transcription_service.storage_service.save_notes(
                        media_filename=storage_name,
                        notes_text=notes_text
                    )
                    logger.info(f"Notes saved to: {notes_path}")
            except Exception as e:
                logger.warning(f"Notes generation failed (non-fatal): {e}")

        response_data["notes"] = notes_text
```

**Step 8: Update the root endpoint's endpoint description**

In the `/` root endpoint (around line 791), update `llm_endpoints` to mention notes:

```python
        "llm_endpoints": {
            "description": "Endpoints that format transcription data for LLM consumption",
            "formats": ["simple", "speaker", "structured", "markdown"],
            "features": ["filler word removal", "speaker merging", "clean text output", "structured notes (generate_notes=true)"]
        }
```

**Step 9: Verify the file parses cleanly**

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); import run_llm_api; print('OK')" 2>&1 | head -5
```

Expected: `OK` (may show model loading warnings, that's fine -- as long as it doesn't raise `SyntaxError` or `ImportError`)

**Step 10: Commit**

```bash
git add src/run_llm_api.py
git commit -m "feat: wire generate_notes flag into LLM transcription endpoints"
```

---

## Task 6: Add `.gitkeep` and update `.gitignore` for models directory

The `models/` directory needs to exist for Docker bind-mount but the actual GGUF (~12GB) must not be committed.

**Step 1: Create the placeholder and update .gitignore**

```bash
touch models/.gitkeep
```

Open `.gitignore` and add:

```
# Downloaded ML models
models/*.gguf
models/*.bin
models/*.safetensors
```

**Step 2: Commit**

```bash
git add models/.gitkeep .gitignore
git commit -m "chore: add models directory placeholder, ignore GGUF files"
```

---

## Task 7: Smoke test the full stack

**Step 1: Build and start services**

```bash
docker-compose up --build -d
```

Watch logs for both services starting:

```bash
docker-compose logs -f llama-cpp    # first run: will download 12GB model
docker-compose logs -f yt-llm-service
```

Wait for the llama-cpp service to print: `llama server listening at http://0.0.0.0:8080`

This may take 5-15 minutes on first run (model download).

**Step 2: Verify both services are healthy**

```bash
curl -s http://localhost:8002/health | python3 -m json.tool
curl -s http://localhost:8080/health | python3 -m json.tool
```

Expected: both return `{"status": "ok"}` (llama-cpp) and `{"status": "ok", "service": "yt-llm-transcription", ...}` (yt-llm-service).

**Step 3: Test notes generation with a short YouTube video**

Use any short video (under 5 minutes). Replace the URL below:

```bash
curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple",
    "generate_notes": true
  }' | python3 -m json.tool
```

Expected response structure:

```json
{
  "success": true,
  "text": "...",
  "language": "en",
  "metadata": {...},
  "notes": "# Never Gonna Give You Up\n\n## Overview\n...",
  "error": null
}
```

**Step 4: Verify notes.md was saved to disk**

```bash
ls data/output/
# Should show a directory named after the video title
ls data/output/Never_Gonna_Give_You_Up/
# Should show: transcription.md  notes.md
cat data/output/Never_Gonna_Give_You_Up/notes.md
```

**Step 5: Test graceful degradation (notes service down)**

```bash
docker-compose stop llama-cpp

curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple",
    "generate_notes": true
  }' | python3 -m json.tool
```

Expected: Response contains `"notes": null` but `"success": true` -- transcription still works.

Restart the service when done:

```bash
docker-compose start llama-cpp
```

**Step 6: Test that `generate_notes=false` (default) has zero overhead**

```bash
curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_format": "simple"
  }' | python3 -m json.tool
```

Check logs -- no call to llama-cpp should appear:

```bash
docker-compose logs yt-llm-service | grep -i "llama\|notes" | tail -20
```

Expected: no log lines mentioning llama-cpp for this request.

**Step 7: Final commit**

```bash
git add .
git commit -m "feat: post-processing notes via gpt-oss-20b - complete implementation"
```

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `llama-cpp` container exits immediately | `entrypoint.sh` has Windows line endings | `dos2unix llama-cpp/entrypoint.sh` |
| Model download fails | Missing `HF_TOKEN` in environment | Set `HF_TOKEN` in `.env` |
| `notes: null` even when llama-cpp is running | Cold start: model still loading | Wait for `llama server listening` log line |
| 12GB GGUF download hangs | HuggingFace rate limit or no token | Use `huggingface-cli login` on host first |
| `CUDA out of memory` during notes generation | WhisperX model still in VRAM | Transcription should release GPU before notes; check for model caching |
| `Invalid JSON` from `/v1/chat/completions` | llama-server started without `--jinja` | Verify entrypoint.sh has the `--jinja` flag |
| Notes response is garbled / non-markdown | `--jinja` missing or wrong model file | Confirm `openai_gpt-oss-20b-MXFP4.gguf` is the downloaded file |
