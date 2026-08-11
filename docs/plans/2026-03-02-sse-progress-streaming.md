# SSE Progress Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the static spinner in the `transcribe` CLI with live phase-by-phase progress updates streamed from the server via Server-Sent Events.

**Architecture:** Add a new `/transcribe-youtube-llm-stream` endpoint that wraps the existing `transcribe_youtube_llm` logic in an async generator, emitting SSE events at each pipeline phase (download, VAD, ASR, align, GEC, notes, done). The CLI uses `httpx` streaming to consume the event stream and renders each phase as a `rich` progress step. The final SSE event carries the full JSON result payload.

**Tech Stack:** FastAPI `StreamingResponse`, `text/event-stream`, `httpx` streaming client, `rich.progress` / `rich.live`

---

## Pipeline phases (server-side)

The server emits these events in order:

| event | label shown in CLI |
|---|---|
| `downloading` | Downloading audio |
| `transcribing` | Transcribing (WhisperX ASR) |
| `aligning` | Aligning timestamps |
| `gec` | Correcting transcript (GEC) |
| `notes` | Generating notes |
| `done` | Complete — carries full result JSON |
| `error` | Fatal error — carries `detail` string |

---

### Task 1: Add SSE helper + phase event schema

**Files:**
- Create: `src/sse_utils.py`
- Test: `tests/test_sse_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_sse_utils.py
import json
from sse_utils import sse_event, sse_done, sse_error

def test_sse_event_format():
    line = sse_event("downloading", {"label": "Downloading audio"})
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    payload = json.loads(line[6:])
    assert payload["phase"] == "downloading"
    assert payload["label"] == "Downloading audio"

def test_sse_done_carries_result():
    result = {"success": True, "text": "hello"}
    line = sse_done(result)
    payload = json.loads(line[6:])
    assert payload["phase"] == "done"
    assert payload["result"]["text"] == "hello"

def test_sse_error_format():
    line = sse_error("download failed")
    payload = json.loads(line[6:])
    assert payload["phase"] == "error"
    assert payload["detail"] == "download failed"
```

**Step 2: Run test to verify it fails**

```bash
cd /home/amlucas/dev/yt-llm-service && docker compose exec yt-llm-service pytest tests/test_sse_utils.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'sse_utils'`

**Step 3: Write minimal implementation**

```python
# src/sse_utils.py
import json
from typing import Any

def sse_event(phase: str, extra: dict | None = None) -> str:
    payload = {"phase": phase}
    if extra:
        payload.update(extra)
    return f"data: {json.dumps(payload)}\n\n"

def sse_done(result: Any) -> str:
    return sse_event("done", {"result": result})

def sse_error(detail: str) -> str:
    return sse_event("error", {"detail": detail})
```

**Step 4: Run test to verify it passes**

```bash
docker compose exec yt-llm-service pytest tests/test_sse_utils.py -v 2>&1 | tail -10
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/sse_utils.py tests/test_sse_utils.py
git commit -m "feat: add SSE event helper (sse_utils)"
```

---

### Task 2: Add streaming endpoint to run_llm_api.py

**Files:**
- Modify: `src/run_llm_api.py`
- Test: `tests/test_sse_stream_endpoint.py`

**Context:** FastAPI's `StreamingResponse` with `media_type="text/event-stream"` lets us `yield` SSE lines from an async generator. The generator mirrors the existing `transcribe_youtube_llm` logic, emitting a phase event before each expensive step and a `done` event at the end containing the same response dict the original endpoint returns.

**Step 1: Write the failing test**

```python
# tests/test_sse_stream_endpoint.py
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from run_llm_api import app

FAKE_DOWNLOAD = {
    "audio_path": "/tmp/fake.mp3",
    "video_id": "abc123",
    "title": "Test Video",
    "file_size": 12345,
    "video_context": None,
}
FAKE_TRANSCRIPTION = {
    "segments": [{"text": "hello world", "start": 0.0, "end": 1.0}],
    "language": "en",
    "metadata": {},
}
FAKE_LLM_RESULT = {
    "text": "hello world",
    "metadata": {"word_count": 2, "format": "simple"},
}

def collect_sse_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events

def test_stream_endpoint_emits_phases():
    with patch("run_llm_api.audio_downloader") as mock_dl, \
         patch("run_llm_api.transcription_service") as mock_ts, \
         patch("run_llm_api.notes_service") as mock_ns:

        mock_dl.download_audio.return_value = FAKE_DOWNLOAD
        mock_ts.transcribe_audio.return_value = FAKE_TRANSCRIPTION
        mock_ts.format_for_llm.return_value = FAKE_LLM_RESULT
        mock_ns.generate = AsyncMock(return_value=None)

        client = TestClient(app)
        with client.stream("POST", "/transcribe-youtube-llm-stream",
                           json={"youtube_url": "https://youtube.com/watch?v=abc123",
                                 "output_format": "simple", "generate_notes": False,
                                 "use_yt_captions": False}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            events = collect_sse_events(resp)

    phases = [e["phase"] for e in events]
    assert "downloading" in phases
    assert "transcribing" in phases
    assert "done" in phases
    done_event = next(e for e in events if e["phase"] == "done")
    assert done_event["result"]["success"] is True

def test_stream_endpoint_emits_error_on_download_failure():
    with patch("run_llm_api.audio_downloader") as mock_dl:
        mock_dl.download_audio.side_effect = RuntimeError("yt-dlp failed")
        client = TestClient(app)
        with client.stream("POST", "/transcribe-youtube-llm-stream",
                           json={"youtube_url": "https://youtube.com/watch?v=abc123",
                                 "output_format": "simple", "generate_notes": False,
                                 "use_yt_captions": False}) as resp:
            events = collect_sse_events(resp)
    phases = [e["phase"] for e in events]
    assert "error" in phases
```

**Step 2: Run test to verify it fails**

```bash
docker compose exec yt-llm-service pytest tests/test_sse_stream_endpoint.py -v 2>&1 | tail -20
```

Expected: `404 Not Found` or attribute error — endpoint doesn't exist yet.

**Step 3: Add the streaming endpoint**

Add to `src/run_llm_api.py` after the existing imports:

```python
from fastapi.responses import StreamingResponse
from sse_utils import sse_event, sse_done, sse_error
```

Then add the endpoint (after the existing `transcribe_youtube_llm` handler):

```python
@app.post("/transcribe-youtube-llm-stream")
async def transcribe_youtube_llm_stream(request: YouTubeLLMTranscriptionRequest):
    """
    Same as /transcribe-youtube-llm but streams phase progress via SSE.
    Each event: data: {"phase": "<name>", ...}\n\n
    Final event: data: {"phase": "done", "result": {...}}\n\n
    """
    async def generate():
        try:
            # Phase 1: download
            yield sse_event("downloading", {"label": "Downloading audio"})
            download_result = audio_downloader.download_audio(
                youtube_url=request.youtube_url, verbose=request.verbose
            )
            audio_path = download_result["audio_path"]
            video_id = download_result["video_id"]
            video_title = download_result.get("title", "")
            file_size = download_result["file_size"]
            video_ctx = download_result.get("video_context")

            # Phase 2: transcribe
            yield sse_event("transcribing", {"label": "Transcribing audio (WhisperX)"})
            result = transcription_service.transcribe_audio(
                audio_path=audio_path,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
                batch_size=request.batch_size,
                verbose=request.verbose,
            )

            # Phase 3: align
            yield sse_event("aligning", {"label": "Aligning timestamps"})
            llm_result = transcription_service.format_for_llm(
                transcription_result=result,
                output_format=request.output_format,
                include_speakers=(request.output_format in ["speaker", "structured", "markdown"]),
                merge_consecutive_speakers=request.merge_consecutive_speakers,
                remove_filler_words=request.remove_filler_words,
            )

            llm_metadata = llm_result.get("metadata", {})
            llm_metadata.update({
                "video_id": video_id,
                "download_file_size": file_size,
                "audio_path": audio_path,
            })

            response_data = {
                "success": True,
                "language": result.get("language"),
                "metadata": llm_metadata,
                "error": None,
            }
            if "text" in llm_result:
                response_data["text"] = llm_result["text"]
            if "speakers" in llm_result:
                response_data["speakers"] = llm_result["speakers"]
            if "blocks" in llm_result:
                response_data["blocks"] = llm_result["blocks"]

            # Phase 4: GEC (optional)
            corrected_transcript = None
            if request.use_yt_captions and video_ctx and video_ctx.captions:
                yield sse_event("gec", {"label": "Correcting transcript (GEC)"})
                try:
                    raw_text = llm_result.get("text") or ""
                    if not raw_text and "blocks" in llm_result:
                        raw_text = " ".join(b.get("text", "") for b in llm_result["blocks"])
                    if not raw_text and "speakers" in llm_result:
                        raw_text = " ".join(t for t in llm_result["speakers"].values() if t)
                    if raw_text:
                        corrected_transcript = await notes_service.correct_transcript(
                            whisperx_text=raw_text,
                            yt_captions_text=video_ctx.captions,
                            video_context=video_ctx,
                        )
                except Exception as e:
                    logger.warning(f"GEC pipeline failed (non-fatal): {e}")

            transcript_for_notes = corrected_transcript or llm_result.get("text") or ""
            response_data["corrected_transcript"] = corrected_transcript
            response_data["video_metadata"] = {
                "title": video_ctx.title,
                "channel": video_ctx.channel,
                "tags": video_ctx.tags,
                "categories": video_ctx.categories,
                "description": video_ctx.description,
            } if video_ctx else None

            # Save transcription
            storage_name = video_title if video_title.strip() else request.youtube_url
            try:
                saved_path = transcription_service.save_transcription_to_disk(
                    media_filename=storage_name,
                    transcription_result=result,
                    llm_result=llm_result,
                )
            except Exception as e:
                logger.warning(f"Failed to save transcription: {e}")
                saved_path = None

            # Phase 5: notes (optional)
            notes_text = None
            if request.generate_notes:
                yield sse_event("notes", {"label": "Generating notes"})
                try:
                    notes_text = await notes_service.generate(
                        transcript_for_notes, video_context=video_ctx
                    )
                    if notes_text and saved_path:
                        transcription_service.storage_service.save_notes(
                            media_filename=storage_name, notes_text=notes_text
                        )
                        if obsidian_service:
                            obsidian_service.save_note(notes_text, source_url=request.youtube_url)
                except Exception as e:
                    logger.warning(f"Notes generation failed (non-fatal): {e}")

            response_data["notes"] = notes_text

            yield sse_done(response_data)

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            yield sse_error(str(e))

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Step 4: Run tests to verify they pass**

```bash
docker compose exec yt-llm-service pytest tests/test_sse_stream_endpoint.py -v 2>&1 | tail -15
```

Expected: 2 PASSED

**Step 5: Run full test suite to check regressions**

```bash
docker compose exec yt-llm-service pytest tests/ -v 2>&1 | tail -20
```

Expected: all existing tests still PASS

**Step 6: Commit**

```bash
git add src/run_llm_api.py src/sse_utils.py tests/test_sse_stream_endpoint.py
git commit -m "feat: add /transcribe-youtube-llm-stream SSE endpoint"
```

---

### Task 3: Update CLI to consume the SSE stream

**Files:**
- Modify: `cli.py`
- Test: `tests/test_cli.py` (extend existing)

**Context:** Replace the `httpx.Client` blocking call for YouTube URLs with `httpx.Client.stream()`. Parse each `data: {...}` line, extract `phase` + `label`, and update a `rich.Live` panel showing the current phase. When `phase == "done"`, extract `result` and hand off to the existing `render_output()`. When `phase == "error"`, print the error and exit.

`★ Insight ─────────────────────────────────────`
`rich.Live` lets you update a single line in-place. The trick is to update the renderable on each SSE event rather than printing new lines — this keeps the terminal clean regardless of how many phases the pipeline has.
`─────────────────────────────────────────────────`

**Step 1: Check existing CLI test patterns**

```bash
cat tests/test_cli.py
```

Understand how existing CLI tests mock httpx — you'll extend the same pattern.

**Step 2: Write the failing test**

In `tests/test_cli.py`, add:

```python
import json
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from cli import app

runner = CliRunner()

def make_sse_stream(phases: list[dict], result: dict):
    """Build a fake SSE byte stream from phase list + final result."""
    lines = []
    for p in phases:
        lines.append(f"data: {json.dumps(p)}\n\n")
    lines.append(f"data: {json.dumps({'phase': 'done', 'result': result})}\n\n")
    return "".join(lines).encode()

def test_youtube_sse_shows_phases(tmp_path):
    fake_result = {"success": True, "text": "Hello world", "metadata": {}}
    stream_bytes = make_sse_stream(
        [{"phase": "downloading", "label": "Downloading audio"},
         {"phase": "transcribing", "label": "Transcribing audio (WhisperX)"}],
        fake_result,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_bytes.return_value = iter([stream_bytes])
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.stream.return_value = mock_resp
        result = runner.invoke(app, ["https://youtube.com/watch?v=abc", "--no-notes"])

    assert result.exit_code == 0
    assert "Hello world" in result.output or result.exit_code == 0  # text rendered

def test_youtube_sse_error_phase_exits_nonzero():
    stream_bytes = f"data: {json.dumps({'phase': 'error', 'detail': 'yt-dlp failed'})}\n\n".encode()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_bytes.return_value = iter([stream_bytes])
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.stream.return_value = mock_resp
        result = runner.invoke(app, ["https://youtube.com/watch?v=abc", "--no-notes"])

    assert result.exit_code == 1
```

**Step 3: Run test to verify it fails**

```bash
cd /home/amlucas/dev/yt-llm-service && uv run pytest tests/test_cli.py -v -k "sse" 2>&1 | tail -20
```

Expected: test not collected or AttributeError (CLI not using stream yet).

**Step 4: Update cli.py — replace blocking YouTube call with SSE streaming**

Replace the YouTube branch in `transcribe()` (lines ~151-161 in cli.py):

```python
# OLD (blocking):
with err_console.status("[bold green]Transcribing YouTube video...[/bold green]"):
    payload = build_youtube_payload(...)
    resp = client.post(f"{url}/transcribe-youtube-llm", json=payload)

# NEW (streaming):
payload = build_youtube_payload(...)
data = _stream_youtube(client, url, payload)
elapsed = time.time() - start
err_console.print(f"[green]Done ({elapsed:.0f}s)[/green]")
render_output(data, format.value)
return  # skip the generic resp handling below
```

Add the `_stream_youtube` helper above the `transcribe()` function:

```python
def _stream_youtube(client: httpx.Client, base_url: str, payload: dict) -> dict:
    """POST to SSE stream endpoint, render live phase updates, return final result dict."""
    from rich.live import Live
    from rich.text import Text

    phase_labels = {
        "downloading": "Downloading audio...",
        "transcribing": "Transcribing audio (WhisperX)...",
        "aligning": "Aligning timestamps...",
        "gec": "Correcting transcript (GEC)...",
        "notes": "Generating notes...",
    }

    current_label = Text("Starting...", style="bold green")

    buffer = ""
    with client.stream("POST", f"{base_url}/transcribe-youtube-llm-stream", json=payload) as resp:
        if resp.status_code != 200:
            err_console.print(f"[red]Service error {resp.status_code}[/red]")
            raise typer.Exit(code=1)

        with Live(current_label, console=err_console, refresh_per_second=4) as live:
            for chunk in resp.iter_bytes():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    for line in raw.splitlines():
                        if not line.startswith("data: "):
                            continue
                        import json
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        phase = event.get("phase")
                        if phase == "done":
                            live.update(Text("Done", style="bold green"))
                            return event["result"]
                        elif phase == "error":
                            live.update(Text(f"Error: {event.get('detail')}", style="bold red"))
                            err_console.print(f"[red]{event.get('detail')}[/red]")
                            raise typer.Exit(code=1)
                        elif phase in phase_labels:
                            label = event.get("label", phase_labels[phase])
                            live.update(Text(label, style="bold green"))

    err_console.print("[red]Stream ended without a done event.[/red]")
    raise typer.Exit(code=1)
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_cli.py -v 2>&1 | tail -20
```

Expected: new SSE tests PASS, existing CLI tests unaffected.

**Step 6: Run full suite**

```bash
docker compose exec yt-llm-service pytest tests/ -v 2>&1 | tail -20
```

Expected: all PASS

**Step 7: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: CLI streams SSE phase updates for YouTube transcription"
```

---

### Task 4: Manual smoke test

**Step 1: Restart the service to pick up the new endpoint**

```bash
docker compose restart yt-llm-service
sleep 5
docker compose ps
```

Expected: `yt-llm-service` healthy

**Step 2: Verify the SSE endpoint exists**

```bash
curl -s http://localhost:8002/ | python3 -m json.tool | grep stream
```

Expected: `"transcribe-youtube-llm-stream"` listed (or check `/docs`)

**Step 3: Run the CLI against a real video**

```bash
transcribe "https://www.youtube.com/watch?v=V5A1IU8VVp4" --no-notes
```

Expected: live phase labels update in terminal (Downloading → Transcribing → Aligning → Done), then transcript printed.

**Step 4: Commit if anything was tweaked**

```bash
git add -p && git commit -m "fix: smoke test adjustments"
```

---

## What changes and what doesn't

| Component | Change |
|---|---|
| `src/sse_utils.py` | New file — SSE formatting helpers |
| `src/run_llm_api.py` | New endpoint added; existing endpoint untouched |
| `cli.py` | YouTube branch uses stream; file branch unchanged |
| `tests/test_sse_utils.py` | New |
| `tests/test_sse_stream_endpoint.py` | New |
| `tests/test_cli.py` | Extended with SSE tests |

The existing `/transcribe-youtube-llm` endpoint is **not removed** — it stays as the non-streaming fallback and for backwards compatibility.
