# Plan: YouTube Captions GEC Correction + Codex Bug Fixes

## Task Description

Two work items:

1. **Transcript correction via YouTube captions (GEC)** — when a YouTube URL is
   transcribed, opportunistically fetch YouTube's own auto-captions (`en-orig`) in
   parallel with audio download. Feed both the WhisperX transcript and the YT captions
   to gpt-oss-20b as a Generative Error Correction (GEC) pass before notes generation.
   The corrected transcript is what gets passed to `NotesService.generate()` and saved
   to disk.

2. **Fix three bugs** identified by Codex review (independent of the above, safe to
   ship first):
   - (HIGH) `user_config.py` crashes on TOML with wrong value types (e.g. `vault_path = 123`)
   - (MEDIUM) `saved_path` unbound in three route handlers when disk-save fails
   - (LOW) Config parse failures log "disabled" masking the real error

## Objective

- Improve word-level accuracy of YouTube transcripts by cross-referencing two
  independent ASR systems (WhisperX + YouTube/Google ASR) whose errors don't overlap
- No model swap, no new Docker image layers, no new service
- Correction is opt-in (`use_yt_captions: bool = True` on the request) and degrades
  gracefully when captions are unavailable
- Only affects YouTube URL endpoints — file uploads are unchanged
- Fix all three Codex bugs so no config mistake can crash startup

## Problem Statement

### Why dual-ASR correction works

WhisperX (`whisper-large-v3-turbo`) and YouTube's ASR (Google) are independent models
trained on different data with different architectures. They make *different* errors on
the same audio. When both agree on a word, it is almost certainly correct. When they
disagree, an LLM with full sentence context can pick the right word — something neither
system can do alone.

Observed from the test video (`3wglqgskzjQ`):
```
YouTube:   "entropy has released..."    ← wrong
WhisperX:  [likely "Anthropic"]         ← possibly right
YouTube:   "two calling"                ← wrong ("tool calling")
YouTube:   "longunning"                 ← wrong ("long-running")
```

A correction prompt that receives both versions as input can resolve these with
context clues.

### Why this is better than a single-transcript correction pass

A single-transcript LLM correction pass has a documented failure mode: the model
cannot know if an unusual word is an error or intentional jargon (e.g. "llama.cpp").
The second transcript provides an explicit signal: if both disagree on a word, at
least one is wrong.

### Bugs

```
user_config.py:58   Path(vault_raw)  — crashes on non-string TOML value
run_llm_api.py:322  if notes_text and saved_path  — UnboundLocalError when save failed
run_llm_api.py:49   "disabled (no config...)"  — hides parse errors
```

## Solution Approach

### Phase 1 — Bug fixes (no functional change, ship immediately)

Three targeted edits, no new dependencies.

### Phase 2 — Caption fetcher in AudioDownloader

Add `fetch_captions(youtube_url) -> Optional[str]` to `AudioDownloader`. Uses yt-dlp
Python API (already imported in `_extract_video_title`) to download `en-orig` VTT,
parse it to plain text, and return it. Returns `None` if unavailable — no exception.

**Critical constraints proven by live container test:**
- Do NOT pass `impersonate: "chrome-131"` in the Python API — it raises `AssertionError`
  in yt-dlp 2026.02.21 (the container version). The subprocess CLI supports it; the
  Python API does not. Use cookies-only or no-auth fallback.
- `subtitleslangs: ["en-orig", "en"]` downloads **both** tracks, not just the first.
  Prefer `en-orig` explicitly: glob for `*.en-orig.vtt` first, fall back to `*.en.vtt`.
- Add `noplaylist: True` to prevent playlist expansion on playlist URLs.
- `process_info()` after `extract_info(download=False)` works correctly for
  subtitle-only download (validated in container).

### Phase 3 — GEC method in NotesService

Add `correct_transcript(whisperx_text, yt_captions_text) -> str` to `NotesService`.
Calls the same llama-cpp endpoint with a constrained correction prompt. Returns the
corrected transcript string (or the original whisperx_text if the call fails).

**Critical constraint:** Do NOT truncate the input before passing to the caller.
The correction method must return a full-length corrected transcript. If the transcript
exceeds the context window, apply segment-level correction (split into ~2000-word
chunks, correct each, rejoin) rather than silently dropping the middle. The caller
uses this text for both notes generation AND disk storage — a truncated return would
be a silent data-loss regression.

### Phase 4 — Update response model, then wire handler

**Response model must be updated before wiring** or FastAPI silently drops the new
field. Add `corrected_transcript: Optional[str] = None` to `LLMTranscriptionResponse`
(src/run_llm_api.py:131).

In `run_llm_api.py`, in the `transcribe_youtube_llm` handler only:
- Caption fetch is sequential (after audio download), not parallel — the handler is
  async but both calls are I/O-bound and the complexity of `asyncio.gather` is not
  warranted here
- After WhisperX transcription and formatting: if `use_yt_captions=True` and captions
  were fetched, run correction pass on the plain-text transcript
- Extract plain text from `llm_result` using all three format fallbacks (text →
  blocks → speakers) before passing to correction — not just `llm_result["text"]`
- Use corrected transcript for notes generation and disk storage
- Store `corrected_transcript` in response

## Relevant Files

- `src/audio_downloader.py` — add `fetch_captions()` and `_parse_vtt()` methods
- `src/notes_service.py` — add `correct_transcript()` method + correction prompt constant
- `src/run_llm_api.py` — (1) add `corrected_transcript` to `LLMTranscriptionResponse`;
  (2) add `use_yt_captions` to `YouTubeLLMTranscriptionRequest`; (3) wire correction
  into `transcribe_youtube_llm`; (4) fix 3 Codex bugs
- `src/user_config.py` — bug fix only
- `tests/test_user_config.py` — add type-crash regression test

### New Files
None.

## Implementation Phases

### Phase 1: Bug Fixes

Fix the three Codex issues. Commit separately before touching anything else.

### Phase 2: Caption Fetcher

Add `fetch_captions()` to `AudioDownloader`. This is isolated — no callers yet.
Test manually in the container before wiring up.

### Phase 3: GEC Correction Method

Add `correct_transcript()` to `NotesService`. Design the correction prompt carefully.
Test manually with the sample VTT already downloaded.

### Phase 4: Update Response Model, Then Wire Handler

**Update `LLMTranscriptionResponse` and `YouTubeLLMTranscriptionRequest` first.**
FastAPI drops undeclared response fields silently — this must come before handler
wiring or the `corrected_transcript` field will be invisible in the response.
Then connect phases 2 and 3 through the handler and run end-to-end test.

## Step by Step Tasks

### 1. Fix HIGH bug — user_config.py type crash

In `src/user_config.py`, move the `UserConfig(...)` construction inside the existing
`try/except` block and coerce `vault_raw` to `str`:

```python
try:
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    obsidian = data.get("obsidian", {})
    vault_raw = str(obsidian.get("vault_path", "~/Documents/obsidian"))
    return UserConfig(
        obsidian_enabled=obsidian.get("enabled", False),
        vault_path=Path(vault_raw).expanduser(),
        inbox_dir=obsidian.get("inbox_dir", "Inbox"),
        tags=obsidian.get("tags", ["video-notes"]),
    )
except Exception as e:
    logger.warning(f"Failed to load user config at {config_path}: {e}")
    return None
```

Add regression test to `tests/test_user_config.py`:
```python
def test_load_handles_bad_value_type(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[obsidian]\nenabled = true\nvault_path = 123\n')
    assert load_user_config(config_path=cfg_file) is None
```

### 2. Fix MEDIUM bug — saved_path unbound in three handlers

In `src/run_llm_api.py`, add `saved_path = None` before each `try` block that assigns
it. Three locations:
- `transcribe_audio_llm` (~line 285)
- `transcribe_youtube_llm` (~line 488)
- `transcribe_file_llm` (~line 723)

### 3. Fix LOW bug — misleading disabled log

In `src/run_llm_api.py` (~line 46-49):

```python
if obsidian_service:
    logger.info(f"Obsidian integration enabled → {_user_cfg.vault_path / _user_cfg.inbox_dir}")
elif _user_cfg is None:
    logger.info("Obsidian integration disabled (config absent or unreadable — see warnings above)")
else:
    logger.info("Obsidian integration disabled (enabled = false in config)")
```

Commit these three fixes as a single commit before proceeding.

### 4. Add fetch_captions() to AudioDownloader

In `src/audio_downloader.py`, add a new method after `_extract_video_title`.

**Proven yt-dlp API pattern** (validated live in container, yt-dlp 2026.02.21):

```python
def fetch_captions(self, youtube_url: str) -> Optional[str]:
    """
    Fetch YouTube's auto-generated English captions (en-orig) and return as plain text.

    Uses yt-dlp Python API. Returns None if captions are unavailable or on any error.
    Never raises.

    IMPORTANT: Do NOT pass 'impersonate' to YoutubeDL() — the Python API in yt-dlp
    2026.02.21 raises AssertionError on that option. Use cookies-only auth.
    """
    import tempfile

    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en-orig", "en"],
            "subtitlesformat": "vtt",
            "outtmpl": "%(id)s.%(ext)s",
            "noplaylist": True,
        }
        # Cookies-only auth — no impersonate (crashes Python API in this yt-dlp version)
        cookie_path = Path("/app/cookies.txt")
        if cookie_path.exists() and cookie_path.stat().st_size > 100:
            ydl_opts["cookiefile"] = str(cookie_path)
        # else: no auth — public videos work without auth for subtitle fetch

        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts["paths"] = {"home": tmpdir}

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                ydl.process_info(info)  # triggers subtitle download without audio

            # Prefer en-orig; fall back to en. glob("*.vtt")[0] is nondeterministic
            # when both tracks are downloaded — select explicitly.
            tmppath = Path(tmpdir)
            vtt_file = next(tmppath.glob("*.en-orig.vtt"), None) \
                    or next(tmppath.glob("*.en.vtt"), None)

            if vtt_file is None:
                logger.debug(f"No captions found for {youtube_url}")
                return None

            vtt_text = vtt_file.read_text(encoding="utf-8")

        return self._parse_vtt(vtt_text)

    except Exception as e:
        logger.debug(f"Caption fetch failed (non-fatal): {e}")
        return None

def _parse_vtt(self, vtt_text: str) -> str:
    """
    Convert WebVTT content to plain deduplicated text.
    Strips timestamps, inline timing tags, cue IDs, and duplicate consecutive lines.
    """
    lines = vtt_text.split("\n")
    clean = []
    for line in lines:
        line = line.strip()
        if (not line
                or line.startswith("WEBVTT")
                or line.startswith("Kind:")
                or line.startswith("Language:")):
            continue
        # skip timestamp lines (00:00:00.000 --> 00:00:01.000 ...)
        if re.match(r"^\d{2}:\d{2}", line):
            continue
        # skip numeric-only cue IDs
        if re.match(r"^\d+$", line):
            continue
        # strip inline timing tags like <00:00:01.200><c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line:
            continue
        # deduplicate consecutive identical lines (VTT repeats lines as captions scroll)
        if not clean or clean[-1] != line:
            clean.append(line)
    return " ".join(clean)
```

### 5. Add correct_transcript() to NotesService

In `src/notes_service.py`, add a new correction prompt constant and method.

**Critical design constraint:** The method must return the full-length corrected
transcript, not a truncated version. If the transcript exceeds the model's context
window, use chunked correction (split into ~2000-word chunks, correct each chunk
independently, rejoin). Never truncate and return a partial result — the caller uses
this for both notes AND disk storage.

```python
CORRECTION_SYSTEM_PROMPT = """\
You are correcting a speech-to-text transcript using a second ASR transcript of the same audio as a reference.

Rules — read carefully:
- Fix only words that are clearly misheard, phonetically substituted, or garbled.
- Use the reference transcript to resolve ambiguous words — if one source has a plausible technical term and the other has a nonsense word, prefer the technical term.
- Do NOT paraphrase, reorder, or restructure sentences.
- Do NOT add content that does not appear in either source.
- Do NOT remove content, including repetitions or filler words.
- Preserve speaker intent exactly — including punctuation style and sentence structure.
- The corrected transcript must be approximately the same length as the primary transcript.
- If both sources have the same error, output what makes most sense in context.
- Output only the corrected transcript. No preamble, no explanation.
"""

CHUNK_WORDS = 2000  # words per correction chunk


# Add as a new method on NotesService:
async def correct_transcript(
    self,
    whisperx_text: str,
    yt_captions_text: str,
) -> str:
    """
    Use gpt-oss-20b to correct the WhisperX transcript using YT captions as reference.

    For long transcripts, applies correction in ~2000-word chunks to avoid context
    limits. Always returns full-length text — never truncates. Falls back to the
    original whisperx_text on any error.
    """
    if not whisperx_text or not whisperx_text.strip():
        return whisperx_text
    if not yt_captions_text or not yt_captions_text.strip():
        return whisperx_text

    log_action("Running GEC transcript correction pass")

    wx_words = whisperx_text.split()

    # Short transcript: single-pass correction
    if len(wx_words) <= CHUNK_WORDS:
        return await self._correct_chunk(whisperx_text, yt_captions_text)

    # Long transcript: chunk-level correction
    # Split WhisperX into chunks; use a proportional window of YT captions per chunk
    yt_words = yt_captions_text.split()
    chunks = [wx_words[i:i+CHUNK_WORDS] for i in range(0, len(wx_words), CHUNK_WORDS)]
    corrected_chunks = []

    for i, chunk in enumerate(chunks):
        # Align a proportional YT captions window to this chunk
        frac_start = i / len(chunks)
        frac_end = (i + 1) / len(chunks)
        yt_start = int(frac_start * len(yt_words))
        yt_end = int(frac_end * len(yt_words))
        yt_chunk = " ".join(yt_words[yt_start:yt_end])
        wx_chunk = " ".join(chunk)

        corrected = await self._correct_chunk(wx_chunk, yt_chunk)
        corrected_chunks.append(corrected)

    result = " ".join(corrected_chunks)
    logger.info(f"GEC chunked correction complete ({len(chunks)} chunks, {len(result)} chars)")
    return result

async def _correct_chunk(self, whisperx_chunk: str, yt_chunk: str) -> str:
    """Single-chunk correction call. Falls back to whisperx_chunk on error."""
    user_content = (
        "## WhisperX transcript (primary — correct this):\n"
        f"{whisperx_chunk}\n\n"
        "## YouTube auto-captions (reference — use to resolve ambiguous words):\n"
        f"{yt_chunk}"
    )
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "developer", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    url = f"{self.base_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        content = (
            response.json().get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content if content else whisperx_chunk
    except Exception as e:
        logger.warning(f"GEC chunk correction failed (non-fatal): {e}")
        return whisperx_chunk
```

Key design decisions:
- WhisperX is "primary" — LLM corrects it, not blends both equally
- YT captions are "reference" — resolve ambiguous words only
- Temperature 0.1 — deterministic correction, not creative generation
- `reasoning_effort: low` — word substitution needs no chain-of-thought
- Chunked for long transcripts — never silently drops content
- Falls back to original chunk on any error — fully non-fatal

### 6. Update LLMTranscriptionResponse and YouTubeLLMTranscriptionRequest

**This must be done before wiring the handler.** FastAPI silently drops response
fields not declared in `response_model`.

In `src/run_llm_api.py`:

```python
class LLMTranscriptionResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    speakers: Optional[dict] = None
    blocks: Optional[list] = None
    language: Optional[str] = None
    metadata: dict
    error: Optional[str] = None
    notes: Optional[str] = None
    corrected_transcript: Optional[str] = None   # NEW
```

```python
class YouTubeLLMTranscriptionRequest(BaseModel):
    youtube_url: str
    output_format: str = "simple"
    # ... existing fields ...
    use_yt_captions: bool = True   # NEW — opt-in by default for YouTube requests
```

### 7. Wire correction into transcribe_youtube_llm handler

In `transcribe_youtube_llm`, after WhisperX transcription and `format_for_llm`, add:

```python
# --- GEC: fetch YT captions and correct transcript (non-fatal) ---
corrected_transcript = None
if request.use_yt_captions:
    try:
        yt_captions = audio_downloader.fetch_captions(request.youtube_url)
        if yt_captions:
            # Extract plain text from all possible format shapes
            raw_text = llm_result.get("text") or ""
            if not raw_text and "blocks" in llm_result:
                raw_text = " ".join(b.get("text","") for b in llm_result["blocks"])
            if not raw_text and "speakers" in llm_result:
                raw_text = " ".join(
                    t for t in llm_result["speakers"].values() if t
                )
            if raw_text:
                corrected_transcript = await notes_service.correct_transcript(
                    whisperx_text=raw_text,
                    yt_captions_text=yt_captions,
                )
                logger.info("GEC correction applied to transcript")
    except Exception as e:
        logger.warning(f"GEC pipeline failed (non-fatal): {e}")

# Use corrected transcript downstream if available
transcript_for_notes = corrected_transcript or llm_result.get("text") or ""
response_data["corrected_transcript"] = corrected_transcript
```

Pass `transcript_for_notes` to `notes_service.generate()` instead of extracting
`llm_result.get("text")` inline in the notes block.

## Testing Strategy

**Bug fixes**: Run existing pytest suite:
```bash
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m pytest tests/test_user_config.py -v
```

**Caption fetcher**: Manual test in container:
```bash
docker compose run --rm --entrypoint="" yt-llm-service python3 -c "
import sys; sys.path.insert(0, '/app/src')
from audio_downloader import AudioDownloader
from pathlib import Path
d = AudioDownloader()
result = d.fetch_captions('https://www.youtube.com/watch?v=3wglqgskzjQ')
print(result[:500] if result else 'None')
"
```

**GEC end-to-end**: Transcribe the test video with `use_yt_captions=true` and compare
the `corrected_transcript` field against the raw `text` field. Look specifically for:
- "Anthropic" vs "entropy"
- "tool calling" vs "two calling"
- "long-running" vs "longunning"

## Acceptance Criteria

- [ ] Bug fixes: `load_user_config` with `vault_path = 123` returns `None`, no crash
- [ ] Bug fixes: `saved_path = None` initialised before each of three `try` blocks
- [ ] Bug fixes: startup log correctly says "config absent" vs "enabled=false"
- [ ] `fetch_captions()` returns a non-empty string for a video with captions
- [ ] `fetch_captions()` returns `None` (no exception) for a video without captions
- [ ] `fetch_captions()` does NOT pass `impersonate` to `YoutubeDL()`
- [ ] `fetch_captions()` selects `*.en-orig.vtt` over `*.en.vtt` deterministically
- [ ] `correct_transcript()` returns a string of approximately the same word count as input
- [ ] `correct_transcript()` returns the original `whisperx_text` on LLM failure
- [ ] `correct_transcript()` processes long transcripts in chunks (not truncated)
- [ ] `LLMTranscriptionResponse` has `corrected_transcript: Optional[str]` field
- [ ] `POST /transcribe-youtube-llm` with `use_yt_captions=true` returns a visible
      `corrected_transcript` field (non-null if captions were available)
- [ ] `POST /transcribe-youtube-llm` with `use_yt_captions=false` skips correction
      and returns `"corrected_transcript": null`
- [ ] Service remains healthy after a correction-pass run
- [ ] Notes generation uses the corrected transcript when available

## Validation Commands

```bash
# 1. Compile check
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m py_compile src/user_config.py src/run_llm_api.py \
    src/audio_downloader.py src/notes_service.py

# 2. Unit tests (bug fixes)
docker compose run --rm --entrypoint="" yt-llm-service \
  python -m pytest tests/test_user_config.py -v

# 3. Caption fetcher smoke test
docker compose run --rm --entrypoint="" yt-llm-service python3 -c "
import sys; sys.path.insert(0, '/app/src')
from audio_downloader import AudioDownloader
d = AudioDownloader()
r = d.fetch_captions('https://www.youtube.com/watch?v=3wglqgskzjQ')
print('OK:', len(r), 'chars') if r else print('FAIL: None returned')
"

# 4. End-to-end correction test
curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=3wglqgskzjQ",
       "output_format":"simple","use_yt_captions":true,"generate_notes":false}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('=== RAW ===')
print(d.get('text','')[:300])
print('=== CORRECTED ===')
print((d.get('corrected_transcript') or 'None')[:300])
"

# 5. Service health check
curl -s http://localhost:8002/health
```

## Notes

- **`en-orig` vs `en`**: `en-orig` is YouTube's primary ASR track. `en` is sometimes
  a translated/adapted track. yt-dlp with `subtitleslangs: ["en-orig", "en"]` downloads
  **both** when available — explicit glob selection (`*.en-orig.vtt` then `*.en.vtt`)
  is required to pick the right one deterministically.
- **`impersonate` crashes the Python API**: Validated live in container (yt-dlp
  2026.02.21). The `impersonate` option is only safe in the subprocess CLI (`yt-dlp`
  command). The existing `_extract_video_title` method has the same latent bug
  at `audio_downloader.py:86` — do not replicate it in `fetch_captions`.
- **Latency budget**: Caption download ~1s. Correction LLM pass: ~2-5 min for a
  60-min video at 20B/MXFP4 on RTX 3090 Ti. For short tech videos (10-20 min), expect
  30-90 seconds. `use_yt_captions=false` skips this entirely.
- **Both transcripts have errors**: This is expected and is exactly why the approach
  works. The correction prompt instructs the LLM to use context to resolve disagreements
  — not to blindly trust either source.
- **File uploads are unchanged**: `fetch_captions` is only called in the YouTube URL
  handler. The file upload endpoint has no YouTube URL to fetch captions for.
- **Old plan superseded**: `specs/parakeet-migration-and-bugfixes.md` is superseded
  by this plan. The Parakeet model migration remains a valid future option if GEC
  correction proves insufficient — it is not in scope here.
- **No new requirements**: yt-dlp is already installed and used. httpx is already used
  by NotesService. No new packages needed.
