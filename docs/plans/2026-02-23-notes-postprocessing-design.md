# Design: Post-Processing Transcription into Structured Notes

**Date:** 2026-02-23
**Status:** Approved

---

## Summary

Add an optional `generate_notes` flag to the existing LLM transcription endpoints. When enabled, the service sends the completed transcript to a local gpt-oss-20b model (via llama.cpp HTTP sidecar) and returns a structured markdown document: title, overview, topical sections with inline quotes. Notes are saved alongside the transcript and returned in the API response.

---

## Architecture Overview

Two Docker services share the 24GB GPU (GPU 0):

- **`yt-llm-service`** (existing) -- WhisperX transcription, FastAPI on port 8002
- **`llama-cpp`** (new) -- gpt-oss-20b MXFP4 GGUF, llama-server OpenAI-compatible API on port 8080

Both pin to `CUDA_VISIBLE_DEVICES=0`. GPU use is sequential: WhisperX completes before notes generation begins. Peak VRAM stays under 20GB (WhisperX ~6-8GB + gpt-oss-20b ~12GB when active), well within the 24GB card.

### Docker Compose topology

```
Host
├── yt-llm-service (port 8002)  → GPU 0
└── llama-cpp       (port 8080)  → GPU 0
    └── model volume: ./models/openai_gpt-oss-20b-MXFP4.gguf
```

---

## Model

- **Model:** `openai/gpt-oss-20b` (OpenAI open-weight, Apache 2.0)
- **GGUF source:** `bartowski/openai_gpt-oss-20b-GGUF`
- **Quantization:** MXFP4 (`openai_gpt-oss-20b-MXFP4.gguf`, ~12GB)
- **Context window:** 128k tokens
- **Chat template:** Harmony format (required for gpt-oss models)
- **Download:** Automated at llama-cpp service startup via `huggingface-cli`

---

## Components

### New files

| File | Purpose |
|------|---------|
| `src/notes_service.py` | `NotesService` class -- calls llama-cpp HTTP, builds Harmony prompt, returns markdown |
| `llama-cpp/Dockerfile` | llama-cpp service image with CUDA support |
| `llama-cpp/entrypoint.sh` | Downloads GGUF if absent, starts `llama-server` |
| `docs/plans/2026-02-23-notes-postprocessing-design.md` | This file |

### Modified files

| File | Change |
|------|--------|
| `docker-compose.yml` | Add `llama-cpp` service, model bind-mount volume |
| `src/run_llm_api.py` | Add `generate_notes: bool = False` to `LLMTranscriptionRequest` and `YouTubeLLMTranscriptionRequest`; call `NotesService` in the two LLM endpoints |
| `src/storage_service.py` | Add `save_notes(directory, notes_text)` method, saves `notes.md` |
| `.env.example` | Add `LLAMA_CPP_URL`, `LLAMA_CPP_GPU_LAYERS`, `NOTES_MAX_TOKENS` |
| `src/config.py` | Add `LLAMA_CPP_URL`, `LLAMA_CPP_GPU_LAYERS`, `NOTES_MAX_TOKENS` config fields |

---

## Data Flow

```
POST /transcribe-youtube-llm
  Body: { youtube_url, output_format, generate_notes: true, ... }

  1. AudioDownloader.download_audio(youtube_url)
  2. TranscriptionService.transcribe_audio(audio_path)
  3. TranscriptionService.format_for_llm(result, output_format)
  4. StorageService.save_transcription(title, transcript, metadata)
      → data/output/<title>/transcription.md
  5. [if generate_notes=true]
     NotesService.generate(transcript_text)
       → POST http://llama-cpp:8080/v1/chat/completions
       → Harmony system prompt + transcript as user message
       → Returns structured markdown string
     StorageService.save_notes(directory, notes_text)
       → data/output/<title>/notes.md
  6. Response: {
       success, text, language, metadata,
       notes: "<markdown>" | null
     }
```

---

## Notes Output Format

The `NotesService` instructs gpt-oss-20b to produce:

```markdown
# <Title inferred from content>

## Overview
<2-3 sentence summary of the video>

## <Topic Section 1>
<Narrative paragraph>
> "Relevant verbatim quote from transcript"

## <Topic Section 2>
...
```

---

## Harmony Prompt Structure

gpt-oss models require the Harmony chat template. The system prompt is sent as a `developer` role message:

```
<|start|>developer<|message|>
You are a note-taking assistant. Given a video transcript, produce a structured
markdown document with: a title, an overview section, and topical sections each
containing a brief narrative and one relevant verbatim quote. Output only the
markdown document, no preamble.
<|end|>

<|start|>user<|message|>
<transcript text here>
<|end|>
```

---

## Error Handling

| Scenario | Behavior |
|----------|---------|
| llama.cpp service unavailable | `NotesService` catches `ConnectionError`/timeout, returns `None`. Endpoint logs warning, returns `notes: null`. Transcription still succeeds. |
| Model not downloaded at startup | `llama-cpp/entrypoint.sh` downloads via `huggingface-cli` before starting server. First startup may take several minutes. |
| Transcript exceeds `NOTES_MAX_TOKENS` (default 100k) | Truncate from the middle, preserve first and last 25% of text. Add notice in prompt. |
| llama.cpp returns malformed/empty response | `NotesService` raises `RuntimeError`, endpoint catches it, returns `notes: null`. |
| `generate_notes=false` (default) | `NotesService` is never called. Zero overhead. |

---

## Configuration

New environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_URL` | `http://llama-cpp:8080` | Base URL for llama.cpp OpenAI-compat API |
| `LLAMA_CPP_GPU_LAYERS` | `99` | Number of model layers to offload to GPU (-1 = all) |
| `NOTES_MAX_TOKENS` | `100000` | Max transcript tokens before truncation |

---

## Testing Plan

**Manual smoke tests:**
1. `docker-compose up --build` -- verify both services start
2. `curl http://localhost:8080/health` -- verify llama.cpp responds
3. `curl -X POST localhost:8002/transcribe-youtube-llm -d '{"youtube_url": "...", "generate_notes": true, "output_format": "simple"}'` -- verify `notes` field in response
4. Check `data/output/<title>/notes.md` exists with correct markdown structure

**Negative-path manual tests:**
- Send `generate_notes: true` while llama.cpp container is stopped -- verify transcription still returns, `notes: null`
- Send `generate_notes: false` -- verify no call to llama.cpp is made (check logs)

---

## Constraints and Assumptions

- GPU 0 (24GB) hosts both services; GPU 1 (8GB) is not used (gpt-oss-20b is too large for 8GB)
- Sequential GPU use is enforced by the application flow -- no concurrent transcription + notes
- The llama.cpp service is not exposed externally (internal Docker network only)
- No automated test suite is added (consistent with existing codebase)
- Model download (~12GB) happens at first `docker-compose up` and is cached in the bind-mount volume
