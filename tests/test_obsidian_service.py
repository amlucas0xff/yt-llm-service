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
