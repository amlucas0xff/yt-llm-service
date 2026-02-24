# YouTube Metadata Context Extraction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate all yt-dlp `extract_info()` calls into a single `get_video_context()` method that returns a `VideoContext` dataclass, then thread that context into the GEC correction prompt and the API response.

**Architecture:** A new `VideoContext` dataclass in `audio_downloader.py` captures title, description, channel, tags, categories, and captions in one network call. `notes_service.correct_transcript()` accepts an optional `VideoContext` and injects a context block into the GEC prompt so the model can preserve domain-specific proper nouns. `LLMTranscriptionResponse` gains a `video_metadata` dict field so callers receive the metadata without a second request.

**Tech Stack:** yt-dlp Python API, FastAPI + Pydantic, Python `dataclasses`, httpx (already used by `notes_service.py`).

**Design doc:** `docs/plans/2026-02-24-youtube-metadata-context-design.md`

---

## Critical Auth Note

`fetch_captions()` and the new `get_video_context()` use the **Python API** (`yt_dlp.YoutubeDL`).
The Python API in yt-dlp 2026.02.21 crashes with `AssertionError` if `impersonate` is passed as an option.
Auth strategy for the Python API: cookies file only — no `impersonate`.
The CLI path (`download_audio()` via subprocess) continues to use `--impersonate chrome-131` as fallback.

---

## Task 1: Add `VideoContext` dataclass and `get_video_context()` to `audio_downloader.py`

**Files:**
- Modify: `src/audio_downloader.py`
- Test: `tests/test_audio_downloader.py` (create if absent)

**Step 1: Write the failing test**

Create `tests/test_audio_downloader.py`:

```python
"""Unit tests for AudioDownloader.get_video_context()"""
from unittest.mock import patch, MagicMock
from audio_downloader import AudioDownloader, VideoContext

FAKE_INFO = {
    "id": "abc123",
    "title": "Claude 3.7 Sonnet Deep Dive",
    "description": "Anthropic releases Claude 3.7 Sonnet with extended thinking.",
    "channel": "Anthropic",
    "uploader": "Anthropic",
    "tags": ["claude", "anthropic", "llm"],
    "categories": ["Science & Technology"],
}


def make_downloader(tmp_path):
    with patch("audio_downloader.shutil.which", return_value="/usr/bin/yt-dlp"):
        return AudioDownloader(temp_dir=str(tmp_path))


def test_get_video_context_returns_dataclass(tmp_path):
    dl = make_downloader(tmp_path)
    with patch("yt_dlp.YoutubeDL") as MockYDL:
        instance = MockYDL.return_value.__enter__.return_value
        instance.extract_info.return_value = FAKE_INFO
        instance.process_info.return_value = None

        ctx = dl.get_video_context("https://www.youtube.com/watch?v=abc123")

    assert isinstance(ctx, VideoContext)
    assert ctx.video_id == "abc123"
    assert ctx.title == "Claude 3.7 Sonnet Deep Dive"
    assert ctx.channel == "Anthropic"
    assert ctx.tags == ["claude", "anthropic", "llm"]
    assert ctx.categories == ["Science & Technology"]
    assert "Anthropic releases" in ctx.description


def test_get_video_context_never_raises_on_error(tmp_path):
    dl = make_downloader(tmp_path)
    with patch("yt_dlp.YoutubeDL", side_effect=Exception("network failure")):
        ctx = dl.get_video_context("https://www.youtube.com/watch?v=abc123")

    assert isinstance(ctx, VideoContext)
    assert ctx.title == ""
    assert ctx.captions is None
```

**Step 2: Run test to verify it fails**

```bash
cd /home/amlucas/dev/yt-llm-service
python -m pytest tests/test_audio_downloader.py -v
```

Expected: `ImportError` — `VideoContext` not yet defined.

**Step 3: Add `VideoContext` dataclass and `get_video_context()` to `audio_downloader.py`**

Add after the existing imports at the top of `src/audio_downloader.py`:

```python
from dataclasses import dataclass, field
```

Add the dataclass before the `AudioDownloader` class definition:

```python
@dataclass
class VideoContext:
    """All metadata and captions for a YouTube video, fetched in one extract_info() call."""
    video_id: str = ""
    title: str = ""
    description: str = ""   # truncated to 500 chars
    channel: str = ""
    tags: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    captions: Optional[str] = None   # parsed VTT plain text, or None
```

Add `get_video_context()` as a method on `AudioDownloader` (place it before `download_audio()`):

```python
def get_video_context(self, youtube_url: str) -> "VideoContext":
    """
    Fetch all YouTube metadata and captions in a single extract_info() call.

    Returns a VideoContext dataclass. Never raises — all fields default to
    empty on any error so audio download can proceed regardless.

    Auth: cookies-only (no impersonate — Python API crashes on that option
    in yt-dlp 2026.02.21).
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
        cookie_path = Path("/app/cookies.txt")
        if cookie_path.exists() and cookie_path.stat().st_size > 100:
            ydl_opts["cookiefile"] = str(cookie_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts["paths"] = {"home": tmpdir}

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                ydl.process_info(info)  # triggers subtitle download

            # Parse captions
            tmppath = Path(tmpdir)
            vtt_file = (
                next(tmppath.glob("*.en-orig.vtt"), None)
                or next(tmppath.glob("*.en.vtt"), None)
            )
            captions = self._parse_vtt(vtt_file.read_text(encoding="utf-8")) if vtt_file else None

        description = (info.get("description") or "")[:500]
        return VideoContext(
            video_id=info.get("id") or self._extract_video_id(youtube_url),
            title=info.get("title") or "",
            description=description,
            channel=info.get("channel") or info.get("uploader") or "",
            tags=info.get("tags") or [],
            categories=info.get("categories") or [],
            captions=captions,
        )

    except Exception as e:
        logger.warning(f"get_video_context() failed (non-fatal): {e}")
        try:
            video_id = self._extract_video_id(youtube_url)
        except Exception:
            video_id = ""
        return VideoContext(video_id=video_id)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_audio_downloader.py -v
```

Expected: both tests PASS.

**Step 5: Commit**

```bash
git add src/audio_downloader.py tests/test_audio_downloader.py
git commit -m "feat: add VideoContext dataclass and get_video_context() to AudioDownloader"
```

---

## Task 2: Update `download_audio()` to use `get_video_context()`

**Files:**
- Modify: `src/audio_downloader.py`

Replace the internal `_extract_video_title()` call inside `download_audio()` with `get_video_context()`. The returned `VideoContext` goes into the result dict.

**Step 1: Write the failing test**

Add to `tests/test_audio_downloader.py`:

```python
def test_download_audio_result_includes_video_context(tmp_path):
    """download_audio() result dict must include 'video_context' key with VideoContext."""
    dl = make_downloader(tmp_path)

    # Create a fake mp3 so the file-exists check passes
    fake_mp3 = tmp_path / "abc123.mp3"
    fake_mp3.write_bytes(b"fake")

    with patch.object(dl, "get_video_context") as mock_ctx, \
         patch.object(dl, "_cleanup_old_files"), \
         patch("subprocess.run") as mock_run:

        mock_ctx.return_value = VideoContext(
            video_id="abc123",
            title="Test Video",
            channel="TestChannel",
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = dl.download_audio("https://www.youtube.com/watch?v=abc123")

    assert "video_context" in result
    assert isinstance(result["video_context"], VideoContext)
    assert result["title"] == "Test Video"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_audio_downloader.py::test_download_audio_result_includes_video_context -v
```

Expected: FAIL — `video_context` key missing from result dict.

**Step 3: Update `download_audio()` in `src/audio_downloader.py`**

In the `download_audio()` method, replace this block:

```python
# Extract video title using yt-dlp before downloading
video_title = self._extract_video_title(youtube_url)
logger.info(f"Video title: {video_title}")
```

with:

```python
# Fetch all metadata + captions in one call
ctx = self.get_video_context(youtube_url)
video_title = ctx.title
logger.info(f"Video title: {video_title}")
```

And update the return dict at the bottom of `download_audio()`:

```python
return {
    "audio_path": str(audio_path),
    "video_id": video_id,
    "title": video_title,
    "file_size": file_size,
    "temp_dir": str(self.temp_dir),
    "video_context": ctx,   # <-- add this line
}
```

**Step 4: Run all tests**

```bash
python -m pytest tests/test_audio_downloader.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/audio_downloader.py tests/test_audio_downloader.py
git commit -m "feat: wire get_video_context() into download_audio() result dict"
```

---

## Task 3: Inject `VideoContext` into GEC correction prompt in `notes_service.py`

**Files:**
- Modify: `src/notes_service.py`
- Test: `tests/test_notes_service.py` (create if absent)

**Step 1: Write the failing test**

Create `tests/test_notes_service.py`:

```python
"""Unit tests for NotesService GEC context injection."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from audio_downloader import VideoContext
from notes_service import NotesService
from config import Config


def make_service():
    cfg = MagicMock(spec=Config)
    cfg.LLAMA_CPP_URL = "http://localhost:8080"
    cfg.NOTES_MAX_TOKENS = 8000
    return NotesService(cfg)


@pytest.mark.asyncio
async def test_correct_transcript_injects_context_block():
    """When VideoContext is provided, the user message must contain the context block."""
    svc = make_service()
    ctx = VideoContext(
        video_id="abc123",
        title="Claude 3.7 Deep Dive",
        channel="Anthropic",
        tags=["claude", "anthropic"],
        categories=["Science & Technology"],
        description="Anthropic releases Claude 3.7 Sonnet.",
    )

    captured_payload = {}

    async def fake_post(url, json=None, **kwargs):
        captured_payload.update(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "corrected text"}}]
        }
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await svc.correct_transcript("raw whisperx", "yt captions", video_context=ctx)

    user_msg = captured_payload["messages"][1]["content"]
    assert "Claude 3.7 Deep Dive" in user_msg
    assert "Anthropic" in user_msg
    assert "claude, anthropic" in user_msg


@pytest.mark.asyncio
async def test_correct_transcript_without_context_omits_block():
    """When no VideoContext is provided, no context block should appear in the prompt."""
    svc = make_service()

    captured_payload = {}

    async def fake_post(url, json=None, **kwargs):
        captured_payload.update(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "corrected text"}}]
        }
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await svc.correct_transcript("raw whisperx", "yt captions")

    user_msg = captured_payload["messages"][1]["content"]
    assert "## Video Context" not in user_msg
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_notes_service.py -v
```

Expected: FAIL — `correct_transcript()` does not accept `video_context` kwarg.

**Step 3: Update `notes_service.py`**

Add import at top of `src/notes_service.py`:

```python
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from audio_downloader import VideoContext
```

Add one rule to `CORRECTION_SYSTEM_PROMPT` (insert after the last rule line, before the closing `"""`):

```
- Use the Video Context block (when present) to identify proper nouns, product
  names, and domain-specific terms that must be preserved exactly as written.
```

Add a helper method on `NotesService` (place before `_correct_chunk()`):

```python
def _build_context_block(self, video_context: "VideoContext") -> str:
    """Format VideoContext as a context block for injection into correction prompts."""
    tags_str = ", ".join(video_context.tags) if video_context.tags else ""
    parts = ["## Video Context (use to resolve domain-specific terms):"]
    if video_context.title:
        parts.append(f"Title: {video_context.title}")
    if video_context.channel:
        parts.append(f"Channel: {video_context.channel}")
    if tags_str:
        parts.append(f"Tags: {tags_str}")
    if video_context.description:
        parts.append(f"Description: {video_context.description}")
    return "\n".join(parts)
```

Update `correct_transcript()` signature:

```python
async def correct_transcript(
    self,
    whisperx_text: str,
    yt_captions_text: str,
    video_context: Optional["VideoContext"] = None,
) -> str:
```

Pass `video_context` through to `_correct_chunk()` in both call sites inside `correct_transcript()`:

```python
# Short transcript path:
return await self._correct_chunk(whisperx_text, yt_captions_text, video_context)

# Chunked path:
corrected = await self._correct_chunk(wx_chunk, yt_chunk, video_context)
```

Update `_correct_chunk()` signature and user message construction:

```python
async def _correct_chunk(
    self,
    whisperx_chunk: str,
    yt_chunk: str,
    video_context: Optional["VideoContext"] = None,
) -> str:
    """Single-chunk correction call. Falls back to whisperx_chunk on error."""
    context_block = (
        self._build_context_block(video_context) + "\n\n"
        if video_context else ""
    )
    user_content = (
        f"{context_block}"
        "## WhisperX transcript (primary — correct this):\n"
        f"{whisperx_chunk}\n\n"
        "## YouTube auto-captions (reference — use to resolve ambiguous words):\n"
        f"{yt_chunk}"
    )
    # rest of method unchanged
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_notes_service.py -v
```

Expected: both PASS.

**Step 5: Commit**

```bash
git add src/notes_service.py tests/test_notes_service.py
git commit -m "feat: inject VideoContext into GEC correction prompt"
```

---

## Task 4: Add `video_metadata` to `LLMTranscriptionResponse` and wire YouTube handler

**Files:**
- Modify: `src/run_llm_api.py`
- Test: `tests/test_run_llm_api.py` (create if absent — smoke test only)

**Step 1: Write the failing test**

Create `tests/test_run_llm_api.py`:

```python
"""Smoke tests for run_llm_api response models."""
from run_llm_api import LLMTranscriptionResponse


def test_llm_transcription_response_has_video_metadata_field():
    """LLMTranscriptionResponse must accept and expose video_metadata."""
    resp = LLMTranscriptionResponse(
        success=True,
        metadata={},
        video_metadata={"title": "Test", "channel": "CH", "tags": [], "categories": [], "description": ""},
    )
    assert resp.video_metadata["title"] == "Test"


def test_llm_transcription_response_video_metadata_defaults_none():
    """video_metadata must default to None when not provided."""
    resp = LLMTranscriptionResponse(success=True, metadata={})
    assert resp.video_metadata is None
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_run_llm_api.py -v
```

Expected: FAIL — `video_metadata` field does not exist on `LLMTranscriptionResponse`.

**Step 3: Add field to `LLMTranscriptionResponse` in `src/run_llm_api.py`**

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
    corrected_transcript: Optional[str] = None
    video_metadata: Optional[dict] = None   # <-- add this line
```

**Step 4: Wire `VideoContext` into the YouTube handler**

Locate the `/transcribe-youtube-llm` handler in `src/run_llm_api.py`.

Find the `download_result` call and extract `VideoContext` from it:

```python
download_result = audio_downloader.download_audio(request.youtube_url, verbose=request.verbose)
audio_path = download_result["audio_path"]
video_title = download_result["title"]
video_id = download_result["video_id"]
video_ctx = download_result.get("video_context")   # <-- add this line
```

Update the GEC call to pass `video_context`:

```python
if request.use_yt_captions and video_ctx and video_ctx.captions:
    corrected_transcript = await notes_service.correct_transcript(
        whisperx_text=transcript_text,
        yt_captions_text=video_ctx.captions,
        video_context=video_ctx,            # <-- add this
    )
```

> Note: the handler currently calls `fetch_captions()` separately to get captions.
> Now that `get_video_context()` already fetched them via `download_audio()`, use
> `video_ctx.captions` directly. Remove the separate `audio_downloader.fetch_captions()`
> call from the handler.

Populate `video_metadata` in the response dict:

```python
response_data["video_metadata"] = {
    "title": video_ctx.title,
    "channel": video_ctx.channel,
    "tags": video_ctx.tags,
    "categories": video_ctx.categories,
    "description": video_ctx.description,
} if video_ctx else None
```

**Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add src/run_llm_api.py tests/test_run_llm_api.py
git commit -m "feat: add video_metadata to LLMTranscriptionResponse and wire YouTube handler"
```

---

## Task 5: Compile check and smoke test

**Step 1: Syntax check all modified files**

```bash
python -m py_compile src/audio_downloader.py src/notes_service.py src/run_llm_api.py && echo "OK"
```

Expected: `OK` (no output before it means no syntax errors).

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

**Step 3: Live smoke test against the running service (optional — requires container running)**

```bash
curl -s -X POST http://localhost:8002/transcribe-youtube-llm \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=3wglqgskzjQ", "use_yt_captions": true}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('video_metadata:', json.dumps(d.get('video_metadata'), indent=2))"
```

Expected: `video_metadata` block printed with title, channel, tags, categories, description.

**Step 4: Commit if any fixes needed, then final commit**

```bash
git add -p
git commit -m "fix: <description of any fixups>"
```

---

## Cleanup: Remove now-dead methods (optional, after all tests pass)

`_extract_video_title()` and `fetch_captions()` are now fully replaced by
`get_video_context()`. They can be removed from `audio_downloader.py`.

Check no other callers exist first:

```bash
grep -r "_extract_video_title\|fetch_captions" src/
```

If only defined, not called from outside:

```bash
git add src/audio_downloader.py
git commit -m "refactor: remove _extract_video_title() and fetch_captions() superseded by get_video_context()"
```
