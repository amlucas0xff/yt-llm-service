# CLI Transcription Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `cli.py` at the project root that lets users transcribe YouTube URLs or local files via the running yt-llm-service, with rich progress indicators and markdown-rendered output.

**Architecture:** Single-file Python CLI using `typer` for argument parsing and `rich` for progress spinners and markdown rendering. Detects input type (URL vs local file) automatically. Calls `/transcribe-youtube-llm` or `/transcribe-file-llm` over HTTP using `httpx`. Renders `structured` blocks as formatted markdown in the terminal.

**Tech Stack:** Python 3.12, typer, rich, httpx (already in requirements.txt)

---

## Context: API Response Shapes

The `/transcribe-youtube-llm` and `/transcribe-file-llm` endpoints return:

```json
{
  "success": true,
  "text": "...",           // present for simple/speaker/markdown formats
  "blocks": [...],         // present for structured format — list of {speaker, text} dicts
  "speakers": {...},       // present for speaker format
  "language": "en",
  "metadata": {...},
  "notes": "..."           // present only if generate_notes=true
}
```

For `structured` format, `blocks` is a list of objects like:
```json
{"speaker": "SPEAKER_00", "text": "Hello, welcome to the show."}
```

The CLI must convert `blocks` to markdown before rendering.

---

### Task 1: Add CLI dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add typer and rich to requirements.txt**

Append these two lines after the existing `httpx` entry:

```
typer>=0.12.0
rich>=13.0.0
```

`httpx` is already present — no change needed there.

**Step 2: Install on host**

```bash
pip install typer rich
```

Expected: Both packages install without errors.

**Step 3: Verify imports work**

```bash
python -c "import typer; import rich; import httpx; print('OK')"
```

Expected: prints `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add typer and rich for CLI"
```

---

### Task 2: Write the failing tests for input detection

**Files:**
- Create: `tests/test_cli.py`

**Step 1: Create test file**

```python
"""Tests for cli.py input detection and output rendering logic."""
import pytest


def test_detect_youtube_url():
    from cli import detect_input_type
    assert detect_input_type("https://www.youtube.com/watch?v=abc123") == "youtube"
    assert detect_input_type("https://youtu.be/abc123") == "youtube"


def test_detect_local_file():
    from cli import detect_input_type
    assert detect_input_type("/tmp/video.mp4") == "file"
    assert detect_input_type("./recording.m4a") == "file"


def test_blocks_to_markdown_single_speaker():
    from cli import blocks_to_markdown
    blocks = [{"speaker": "SPEAKER_00", "text": "Hello world."}]
    result = blocks_to_markdown(blocks)
    assert "**SPEAKER_00**" in result
    assert "Hello world." in result


def test_blocks_to_markdown_multiple_speakers():
    from cli import blocks_to_markdown
    blocks = [
        {"speaker": "SPEAKER_00", "text": "First line."},
        {"speaker": "SPEAKER_01", "text": "Second line."},
        {"speaker": "SPEAKER_00", "text": "Third line."},
    ]
    result = blocks_to_markdown(blocks)
    assert result.count("**SPEAKER_00**") == 2
    assert result.count("**SPEAKER_01**") == 1


def test_blocks_to_markdown_no_speaker():
    from cli import blocks_to_markdown
    blocks = [{"text": "No speaker info here."}]
    result = blocks_to_markdown(blocks)
    assert "No speaker info here." in result
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: All 5 tests FAIL with `ModuleNotFoundError: No module named 'cli'`

---

### Task 3: Implement `detect_input_type` and `blocks_to_markdown`

**Files:**
- Create: `cli.py`

**Step 1: Create cli.py with just the two utility functions**

```python
"""
CLI for yt-llm-service — transcribe YouTube URLs or local files.

Usage:
  uv run cli.py "https://youtube.com/watch?v=..."
  uv run cli.py /path/to/video.mp4 --format speaker --notes
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(add_completion=False, help="Transcribe YouTube URLs or local video/audio files.")


class OutputFormat(str, Enum):
    simple = "simple"
    speaker = "speaker"
    structured = "structured"
    markdown = "markdown"


def detect_input_type(input_str: str) -> str:
    """Return 'youtube' if input looks like a URL, 'file' otherwise."""
    if input_str.startswith("http://") or input_str.startswith("https://"):
        return "youtube"
    return "file"


def blocks_to_markdown(blocks: list[dict]) -> str:
    """Convert structured blocks list to a markdown string."""
    lines = []
    for block in blocks:
        speaker = block.get("speaker")
        text = block.get("text", "")
        if speaker:
            lines.append(f"**{speaker}:** {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)
```

**Step 2: Run tests to confirm utility functions pass**

```bash
pytest tests/test_cli.py -v
```

Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add CLI skeleton with input detection and block renderer"
```

---

### Task 4: Write failing tests for HTTP call logic

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Add HTTP tests using httpx mock**

Append to `tests/test_cli.py`:

```python
def test_build_youtube_payload():
    from cli import build_youtube_payload
    payload = build_youtube_payload(
        url="https://youtu.be/abc",
        fmt="structured",
        notes=True,
        no_filler=False,
        min_speakers=None,
        max_speakers=None,
    )
    assert payload["youtube_url"] == "https://youtu.be/abc"
    assert payload["output_format"] == "structured"
    assert payload["generate_notes"] is True
    assert payload["remove_filler_words"] is False
    assert "min_speakers" not in payload  # None values should be excluded


def test_build_file_fields():
    from cli import build_file_fields
    fields = build_file_fields(
        fmt="simple",
        notes=False,
        no_filler=True,
        min_speakers=2,
        max_speakers=4,
    )
    assert fields["output_format"] == "simple"
    assert fields["generate_notes"] == "false"
    assert fields["remove_filler_words"] == "true"
    assert fields["min_speakers"] == "2"
    assert fields["max_speakers"] == "4"
```

**Step 2: Run to confirm new tests fail**

```bash
pytest tests/test_cli.py::test_build_youtube_payload tests/test_cli.py::test_build_file_fields -v
```

Expected: Both FAIL with `ImportError`

---

### Task 5: Implement `build_youtube_payload` and `build_file_fields`

**Files:**
- Modify: `cli.py`

**Step 1: Add the two builder functions to cli.py (after `blocks_to_markdown`)**

```python
def build_youtube_payload(
    url: str,
    fmt: str,
    notes: bool,
    no_filler: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    """Build JSON payload for /transcribe-youtube-llm."""
    payload: dict = {
        "youtube_url": url,
        "output_format": fmt,
        "generate_notes": notes,
        "remove_filler_words": no_filler,
        "merge_consecutive_speakers": True,
    }
    if min_speakers is not None:
        payload["min_speakers"] = min_speakers
    if max_speakers is not None:
        payload["max_speakers"] = max_speakers
    return payload


def build_file_fields(
    fmt: str,
    notes: bool,
    no_filler: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    """Build multipart form fields for /transcribe-file-llm.
    All values must be strings for httpx multipart encoding."""
    fields: dict = {
        "output_format": fmt,
        "generate_notes": "true" if notes else "false",
        "remove_filler_words": "true" if no_filler else "false",
        "merge_consecutive_speakers": "true",
    }
    if min_speakers is not None:
        fields["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        fields["max_speakers"] = str(max_speakers)
    return fields
```

**Step 2: Run all tests**

```bash
pytest tests/test_cli.py -v
```

Expected: All 7 tests PASS

**Step 3: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add payload builders for youtube and file upload endpoints"
```

---

### Task 6: Implement the main `transcribe` command

**Files:**
- Modify: `cli.py`

**Step 1: Add the typer command and output renderer after the builder functions**

```python
def render_output(data: dict, fmt: str) -> None:
    """Print transcription result to stdout."""
    if fmt == "structured" and data.get("blocks"):
        md_text = blocks_to_markdown(data["blocks"])
        console.print(Markdown(md_text))
    elif fmt == "markdown" and data.get("text"):
        console.print(Markdown(data["text"]))
    elif data.get("text"):
        print(data["text"])
    elif data.get("blocks"):
        # Fallback: render blocks even for non-structured formats
        print(blocks_to_markdown(data["blocks"]))
    else:
        err_console.print("[red]No transcription text in response.[/red]")

    if data.get("notes"):
        print("\n--- Notes ---\n")
        console.print(Markdown(data["notes"]))


@app.command()
def transcribe(
    input: str = typer.Argument(..., help="YouTube URL or path to a local file"),
    format: OutputFormat = typer.Option(OutputFormat.structured, "--format", help="Output format"),
    notes: bool = typer.Option(False, "--notes", help="Generate structured notes via LLM"),
    no_filler: bool = typer.Option(False, "--no-filler", help="Remove filler words (um, uh, like...)"),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Minimum number of speakers"),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Maximum number of speakers"),
    url: str = typer.Option("http://localhost:8002", "--url", help="Service base URL"),
) -> None:
    input_type = detect_input_type(input)
    start = time.time()

    try:
        with httpx.Client(timeout=600) as client:
            if input_type == "youtube":
                with err_console.status("[bold green]Transcribing YouTube video...[/bold green]"):
                    payload = build_youtube_payload(
                        url=input,
                        fmt=format.value,
                        notes=notes,
                        no_filler=no_filler,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    resp = client.post(f"{url}/transcribe-youtube-llm", json=payload)
            else:
                file_path = Path(input)
                if not file_path.exists():
                    err_console.print(f"[red]File not found: {input}[/red]")
                    raise typer.Exit(code=1)
                fields = build_file_fields(
                    fmt=format.value,
                    notes=notes,
                    no_filler=no_filler,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                with err_console.status("[bold green]Transcribing file...[/bold green]"):
                    with open(file_path, "rb") as f:
                        resp = client.post(
                            f"{url}/transcribe-file-llm",
                            data=fields,
                            files={"file": (file_path.name, f)},
                        )

        if resp.status_code != 200:
            err_console.print(f"[red]Service error {resp.status_code}: {resp.json().get('detail', resp.text)}[/red]")
            raise typer.Exit(code=1)

        data = resp.json()
        elapsed = time.time() - start
        err_console.print(f"[green]Done ({elapsed:.0f}s)[/green]")
        render_output(data, format.value)

    except httpx.ConnectError:
        err_console.print(f"[red]Cannot connect to service at {url}. Is docker compose up?[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
```

**Step 2: Smoke test the CLI help**

```bash
python cli.py --help
```

Expected: Prints usage with all flags listed, no errors.

**Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: implement transcribe command with rich spinner and markdown output"
```

---

### Task 7: Manual end-to-end test

**Prerequisites:** Docker services must be running (`docker compose up -d`)

**Step 1: Test YouTube URL**

```bash
python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --format structured
```

Expected: Rich spinner appears on stderr, then structured markdown transcript prints to stdout.

**Step 2: Test with --notes flag**

```bash
python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --notes
```

Expected: Transcript followed by `--- Notes ---` section.

**Step 3: Test error handling**

```bash
python cli.py /nonexistent/file.mp4
```

Expected: Red error message "File not found", exit code 1.

**Step 4: Test stdout is pipeable**

```bash
python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>/dev/null | wc -c
```

Expected: Non-zero character count (transcript text only, no spinner noise).

**Step 5: Commit**

```bash
git add cli.py
git commit -m "feat: CLI transcription tool complete"
```

---

## Notes for the implementer

- Spinner uses `err_console` (stderr) so stdout stays clean for piping
- `httpx` timeout is 600s — transcription of long videos can take several minutes
- `structured` blocks rendering: each `{speaker, text}` block becomes `**SPEAKER_XX:** text`
- Notes are always rendered as markdown regardless of `--format`
- The `--url` flag lets users point at a non-default host if needed
