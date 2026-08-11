import pytest
from pathlib import Path
from datetime import date


SAMPLE_NOTES = """# Understanding Transformers

## Overview
This video covers transformer architecture in depth.

## Takeaways
- Attention is all you need
"""


def test_save_note_creates_file(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    path = svc.save_note(SAMPLE_NOTES, source_url="https://youtu.be/abc")
    assert path is not None
    assert Path(path).exists()


def test_save_note_uses_title_as_filename(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    path = svc.save_note(SAMPLE_NOTES, source_url="https://youtu.be/abc")
    assert Path(path).name == "Understanding Transformers.md"


def test_save_note_creates_inbox_dir(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    svc.save_note(SAMPLE_NOTES, source_url="https://youtu.be/abc")
    assert (tmp_path / "Inbox").is_dir()


def test_save_note_has_yaml_frontmatter(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox", tags=["video-notes"])
    path = svc.save_note(SAMPLE_NOTES, source_url="https://youtu.be/abc")
    content = Path(path).read_text()
    assert content.startswith("---\n")
    assert "source: https://youtu.be/abc" in content
    assert "video-notes" in content
    today = date.today().isoformat()
    assert f"date: {today}" in content


def test_save_note_falls_back_to_untitled_when_no_heading(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    result = svc.save_note("No heading here.\n", source_url="https://youtu.be/abc")
    assert result is not None
    assert Path(result).exists()
    assert Path(result).name == "Untitled.md"


def test_save_note_sanitizes_title(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    notes = "# My Video: Part 1/2 — Advanced\n\nContent here."
    path = svc.save_note(notes, source_url="https://youtu.be/abc")
    filename = Path(path).name
    assert ":" not in filename
    assert "/" not in filename


def test_save_note_records_source_transcript_path(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    path = svc.save_note(
        SAMPLE_NOTES,
        source_url="https://youtu.be/abc",
        source_transcript="/app/output/Understanding Transformers/transcription_1.md",
    )
    content = Path(path).read_text()
    assert "source_transcript: /app/output/Understanding Transformers/transcription_1.md" in content


def test_save_note_records_truncation_flag(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")

    truncated_path = svc.save_note(SAMPLE_NOTES, truncated=True)
    assert "truncated: true" in Path(truncated_path).read_text()

    intact_path = svc.save_note(SAMPLE_NOTES, truncated=False)
    assert "truncated: false" in Path(intact_path).read_text()


def test_save_note_omits_source_transcript_when_unknown(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    path = svc.save_note(SAMPLE_NOTES, source_url="https://youtu.be/abc")
    assert "source_transcript:" not in Path(path).read_text()


@pytest.mark.asyncio
async def test_truncation_flag_reflects_real_notes_generation(tmp_path):
    """A transcript long enough to be truncated must land as truncated: true."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from obsidian_service import ObsidianService
    from notes_service import NotesService
    from config import Config

    cfg = MagicMock(spec=Config)
    cfg.LLAMA_CPP_URL = "http://localhost:8080"
    cfg.NOTES_MAX_TOKENS = 10
    notes_svc = NotesService(cfg)

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": SAMPLE_NOTES}}]}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=resp)
        long_result = await notes_svc.generate(" ".join(f"word{i}" for i in range(500)))
        short_result = await notes_svc.generate("tiny")

    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")

    long_path = svc.save_note(long_result.text, truncated=long_result.truncated)
    assert "truncated: true" in Path(long_path).read_text()

    short_path = svc.save_note(short_result.text, truncated=short_result.truncated)
    assert "truncated: false" in Path(short_path).read_text()
