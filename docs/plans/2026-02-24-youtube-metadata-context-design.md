# Design: YouTube Metadata Context Extraction

**Date**: 2026-02-24
**Status**: Approved
**Scope**: `audio_downloader.py`, `notes_service.py`, `run_llm_api.py`

---

## Problem

The current pipeline makes redundant yt-dlp `extract_info()` network calls:

- `_extract_video_title()` — one call for title only
- `fetch_captions()` — one call for VTT subtitles

A third call would be needed to capture description, tags, and categories. No
metadata beyond the title is currently captured or used to improve transcription
quality. The GEC correction pass has no domain-term signal, making it liable to
"correct" proper nouns (e.g., "Anthropic", "Claude") that WhisperX rendered correctly.

---

## Decision

Consolidate all yt-dlp `extract_info()` calls into one `get_video_context(url)`
method on `AudioDownloader` that returns a `VideoContext` dataclass. Thread this
context through to:

1. **GEC correction prompt** — inject a context block so the model knows
   domain-specific proper nouns to preserve
2. **API response** — expose as a `video_metadata` field on `LLMTranscriptionResponse`

---

## Architecture

```
YouTube URL
    |
    v
AudioDownloader.get_video_context(url)
    |  -- single extract_info() call
    |  -- captures: title, description, channel,
    |               tags, categories, captions (VTT)
    |
    returns VideoContext dataclass
    |
    +-------------------------------+
    |                               |
    v                               v
download_audio()           VideoContext.captions --> correct_transcript()
  (returns audio_path)         context block injected into GEC prompt
                                   --> corrected_transcript
                           VideoContext serialized into
                           LLMTranscriptionResponse.video_metadata
```

---

## Components

### 1. `VideoContext` dataclass (`audio_downloader.py`)

```python
@dataclass
class VideoContext:
    video_id: str
    title: str
    description: str      # truncated to 500 chars
    channel: str
    tags: list[str]
    categories: list[str]
    captions: Optional[str]   # parsed VTT plain text, or None
```

`get_video_context(url) -> VideoContext` replaces both `_extract_video_title()`
and `fetch_captions()`. Never raises — all fields default to empty on error.

`download_audio()` calls `get_video_context()` internally and returns the
`VideoContext` in its result dict alongside `audio_path`.

---

### 2. GEC Prompt Context Injection (`notes_service.py`)

`correct_transcript()` and `_correct_chunk()` gain an optional
`video_context: Optional[VideoContext]` parameter.

When present, a context block is prepended to the user message:

```
## Video Context (use to resolve domain-specific terms):
Title: <title>
Channel: <channel>
Tags: <tag1>, <tag2>, ...
Description: <first 500 chars>

## WhisperX transcript (primary — correct this):
...
```

`CORRECTION_SYSTEM_PROMPT` gains one additional rule:

> "- Use the Video Context block to identify proper nouns, product names, and
>    domain-specific terms that must be preserved exactly as written."

---

### 3. `video_metadata` Response Field (`run_llm_api.py`)

`LLMTranscriptionResponse` gains:

```python
video_metadata: Optional[dict] = None
```

Populated in the YouTube handler from `VideoContext`:

```python
"video_metadata": {
    "title": ctx.title,
    "channel": ctx.channel,
    "tags": ctx.tags,
    "categories": ctx.categories,
    "description": ctx.description,
}
```

---

## Data Flow (YouTube handler)

```
POST /transcribe-youtube-llm
    |
    v
get_video_context(url)              <-- single yt-dlp extract_info call
    |
    +-- ctx.captions  ------------> correct_transcript(wx, yt, ctx)
    |                                   |
    |                                   v
    |                               GEC with context block
    |                               --> corrected_transcript
    |
    +-- ctx  ----------------------> video_metadata in response
    |
    +-- ctx.title  ---------------> storage filename
                                    (replaces _extract_video_title() call)
```

---

## Files Changed

| File | Change |
|------|--------|
| `audio_downloader.py` | Add `VideoContext` dataclass + `get_video_context()`. Update `download_audio()` to call it. Remove `_extract_video_title()` and `fetch_captions()` (merged into `get_video_context()`). |
| `notes_service.py` | Add `video_context: Optional[VideoContext]` param to `correct_transcript()` and `_correct_chunk()`. Update `CORRECTION_SYSTEM_PROMPT`. Inject context block into user message when available. |
| `run_llm_api.py` | Add `video_metadata: Optional[dict]` to `LLMTranscriptionResponse`. Pass `VideoContext` to `correct_transcript()`. Use `ctx.title` for storage filename. |

---

## Error Handling

- `get_video_context()` never raises. On any failure: description/tags/categories
  default to empty strings/lists, captions to `None`.
- Audio download proceeds regardless of metadata extraction success.
- Context injection is best-effort: if `VideoContext` has no meaningful metadata,
  GEC falls back to current behavior (captions-only reference).

---

## What Does NOT Change

- File upload endpoints (`/transcribe-file-llm`) — unaffected
- Notes generation (`generate()`) — no context injection
- Storage paths and Obsidian integration — use `ctx.title` same as before
- The `use_yt_captions` request flag — still controls whether GEC runs at all
