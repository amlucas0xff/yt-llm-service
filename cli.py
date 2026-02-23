# /// script
# dependencies = [
#   "typer>=0.12.0",
#   "rich>=13.0.0",
#   "httpx>=0.25.0",
# ]
# ///
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
