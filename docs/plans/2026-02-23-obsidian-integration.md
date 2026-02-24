# Obsidian Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When notes are generated, optionally mirror them to an Obsidian vault inbox with YAML frontmatter — controlled by a user config file at `~/.config/yt-llm/config.toml`.

**Architecture:** A new `ObsidianService` reads the user config on startup. If enabled, after `StorageService.save_notes()` succeeds, `ObsidianService.save_note()` is called with the notes text and source URL. It parses the `# Title` heading, prepends YAML frontmatter, and writes a `.md` file to `{vault_path}/{inbox_dir}/`. If the config file is absent or `enabled = false`, the service is `None` and the write is silently skipped — the main flow is never blocked.

**Tech Stack:** Python 3.11+ stdlib `tomllib` (no new dependency), `pathlib`, FastAPI startup hook

---

### Task 1: UserConfig loader

**Files:**
- Create: `src/user_config.py`
- Test: `tests/test_user_config.py`

**Step 1: Write the failing test**

```python
# tests/test_user_config.py
import pytest
from pathlib import Path


def test_load_returns_none_when_file_missing(tmp_path):
    from user_config import load_user_config
    config = load_user_config(config_path=tmp_path / "nonexistent.toml")
    assert config is None


def test_load_parses_obsidian_section(tmp_path):
    from user_config import load_user_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[obsidian]\nenabled = true\nvault_path = "~/Documents/obsidian"\n'
        'inbox_dir = "Inbox"\ntags = ["video-notes"]\n'
    )
    config = load_user_config(config_path=cfg_file)
    assert config is not None
    assert config.obsidian_enabled is True
    assert config.vault_path == Path.home() / "Documents" / "obsidian"
    assert config.inbox_dir == "Inbox"
    assert config.tags == ["video-notes"]


def test_load_obsidian_disabled_by_default(tmp_path):
    from user_config import load_user_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[obsidian]\nvault_path = \"~/Documents/obsidian\"\n")
    config = load_user_config(config_path=cfg_file)
    assert config.obsidian_enabled is False


def test_load_default_tags_when_absent(tmp_path):
    from user_config import load_user_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[obsidian]\nenabled = true\nvault_path = "~/Documents/obsidian"\n')
    config = load_user_config(config_path=cfg_file)
    assert config.tags == ["video-notes"]
```

**Step 2: Run test to verify it fails**

```bash
cd /home/amlucas/dev/yt-llm-service
uv run pytest tests/test_user_config.py -v
```
Expected: ImportError or ModuleNotFoundError — `user_config` does not exist yet.

**Step 3: Write minimal implementation**

```python
# src/user_config.py
"""
User-level config loader for yt-llm-service.

Config file location: ~/.config/yt-llm/config.toml

Example:
    [obsidian]
    enabled = true
    vault_path = "~/Documents/obsidian"
    inbox_dir = "Inbox"
    tags = ["video-notes", "auto-generated"]
"""

import tomllib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "yt-llm" / "config.toml"


@dataclass
class UserConfig:
    obsidian_enabled: bool
    vault_path: Path
    inbox_dir: str
    tags: list[str]


def load_user_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Optional[UserConfig]:
    """
    Load user config from TOML file.

    Returns None if the file does not exist.
    Returns UserConfig with obsidian_enabled=False if [obsidian] section is missing
    or enabled key is absent/false.
    """
    if not config_path.exists():
        logger.debug(f"No user config found at {config_path}")
        return None

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse user config at {config_path}: {e}")
        return None

    obsidian = data.get("obsidian", {})
    vault_raw = obsidian.get("vault_path", "~/Documents/obsidian")

    return UserConfig(
        obsidian_enabled=obsidian.get("enabled", False),
        vault_path=Path(vault_raw).expanduser(),
        inbox_dir=obsidian.get("inbox_dir", "Inbox"),
        tags=obsidian.get("tags", ["video-notes"]),
    )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_user_config.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add src/user_config.py tests/test_user_config.py
git commit -m "feat: add UserConfig loader for ~/.config/yt-llm/config.toml"
```

---

### Task 2: ObsidianService

**Files:**
- Create: `src/obsidian_service.py`
- Test: `tests/test_obsidian_service.py`

**Step 1: Write the failing tests**

```python
# tests/test_obsidian_service.py
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
    # Notes with no # heading — should save with "Untitled.md" fallback
    result = svc.save_note("No heading here.\n", source_url="https://youtu.be/abc")
    assert result is not None
    assert Path(result).exists()
    assert Path(result).name == "Untitled.md"


def test_save_note_sanitizes_title(tmp_path):
    from obsidian_service import ObsidianService
    svc = ObsidianService(vault_path=tmp_path, inbox_dir="Inbox")
    notes = "# My Video: Part 1/2 — Advanced\n\nContent here."
    path = svc.save_note(notes, source_url="https://youtu.be/abc")
    # Colon, slash, em-dash must not appear in filename
    filename = Path(path).name
    assert ":" not in filename
    assert "/" not in filename
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_obsidian_service.py -v
```
Expected: ImportError — `obsidian_service` does not exist.

**Step 3: Write minimal implementation**

```python
# src/obsidian_service.py
"""
ObsidianService: writes generated notes to an Obsidian vault inbox.

Never raises — all errors are logged and None is returned so the caller
(run_llm_api.py) is never blocked by a failed Obsidian write.
"""

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Characters illegal in most filesystems / Obsidian filenames
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\u2014\u2013]')


class ObsidianService:
    def __init__(
        self,
        vault_path: Path,
        inbox_dir: str = "Inbox",
        tags: list[str] | None = None,
    ):
        self.vault_path = Path(vault_path)
        self.inbox_dir = inbox_dir
        self.tags = tags or ["video-notes"]

    def _extract_title(self, notes_text: str) -> str:
        """Parse the first # heading from notes markdown. Falls back to 'Untitled'."""
        for line in notes_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return "Untitled"

    def _sanitize_filename(self, title: str) -> str:
        """Remove filesystem-illegal characters and trim length."""
        sanitized = _ILLEGAL_CHARS.sub("", title).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized[:100] if sanitized else "Untitled"

    def _build_frontmatter(self, source_url: Optional[str]) -> str:
        lines = [
            "---",
            f"date: {date.today().isoformat()}",
        ]
        if source_url:
            lines.append(f"source: {source_url}")
        if self.tags:
            lines.append("tags:")
            for tag in self.tags:
                lines.append(f"  - {tag}")
        lines.append("---")
        return "\n".join(lines) + "\n\n"

    def save_note(
        self, notes_text: str, source_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Write notes to {vault_path}/{inbox_dir}/{title}.md with YAML frontmatter.

        Returns the absolute path string on success, None on failure.
        """
        try:
            inbox = self.vault_path / self.inbox_dir
            inbox.mkdir(parents=True, exist_ok=True)

            title = self._extract_title(notes_text)
            filename = self._sanitize_filename(title) + ".md"
            file_path = inbox / filename

            frontmatter = self._build_frontmatter(source_url)
            content = frontmatter + notes_text

            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Obsidian note saved: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save Obsidian note: {e}")
            return None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_obsidian_service.py -v
```
Expected: 7 PASSED.

**Step 5: Commit**

```bash
git add src/obsidian_service.py tests/test_obsidian_service.py
git commit -m "feat: add ObsidianService to write notes to vault inbox"
```

---

### Task 3: Wire into FastAPI startup and endpoints

**Files:**
- Modify: `src/run_llm_api.py`

**IMPORTANT — actual structure:** There is no startup/lifespan hook in `run_llm_api.py`. Services are module-level globals initialized at import time (e.g. `notes_service = NotesService(config)` at line 33). Follow exactly the same pattern.

**Step 1: Add imports at the top of run_llm_api.py**

In the imports block alongside existing service imports, add:
```python
from user_config import load_user_config
from obsidian_service import ObsidianService
```

**Step 2: Add module-level global initialization**

Find the block where `notes_service = NotesService(config)` is declared (around line 33) and add immediately after it:
```python
_user_cfg = load_user_config()
obsidian_service: Optional[ObsidianService] = (
    ObsidianService(
        vault_path=_user_cfg.vault_path,
        inbox_dir=_user_cfg.inbox_dir,
        tags=_user_cfg.tags,
    )
    if _user_cfg and _user_cfg.obsidian_enabled
    else None
)
if obsidian_service:
    logger.info(f"Obsidian integration enabled → {_user_cfg.vault_path / _user_cfg.inbox_dir}")
else:
    logger.info("Obsidian integration disabled (no config or enabled=false)")
```

**Step 4: Call save_note after save_notes in each endpoint**

In each endpoint that calls `storage_service.save_notes(...)`, immediately after that call add:
```python
if obsidian_service:
    obsidian_service.save_note(notes, source_url=<url_or_filename>)
```

For YouTube endpoints: `source_url=youtube_url` (or whatever the param is named).
For file upload endpoints: `source_url=None` (no URL available).

There are 3 endpoints — find each `save_notes` call and add the Obsidian call below it.

**Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```
Expected: all existing tests pass (the new Obsidian service is optional and does not break existing behavior).

**Step 6: Commit**

```bash
git add src/run_llm_api.py
git commit -m "feat: wire ObsidianService into FastAPI startup and note-generation endpoints"
```

---

### Task 4: Create example config file + update README

**Files:**
- Create: `config.example.toml`
- Modify: `README.md`

**Step 1: Create the example config**

```toml
# config.example.toml
# User config for yt-llm-service.
# Copy to ~/.config/yt-llm/config.toml and edit as needed.

[obsidian]
# Set enabled = true to mirror generated notes to your Obsidian vault.
enabled = false
vault_path = "~/Documents/obsidian"
inbox_dir = "Inbox"
tags = ["video-notes", "auto-generated"]
```

**Step 2: Add a short section to README.md**

Find the relevant section (likely after the CLI usage section) and add:

```markdown
## Obsidian Integration

Generated notes can be automatically mirrored to your Obsidian vault.

1. Copy the example config:
   ```bash
   mkdir -p ~/.config/yt-llm
   cp config.example.toml ~/.config/yt-llm/config.toml
   ```
2. Edit `~/.config/yt-llm/config.toml`:
   ```toml
   [obsidian]
   enabled = true
   vault_path = "~/Documents/obsidian"   # path to your vault
   inbox_dir = "Inbox"                    # subdirectory inside vault
   tags = ["video-notes"]
   ```
3. Run `transcribe` as usual — if notes are generated, they are also saved to `{vault_path}/{inbox_dir}/{Title}.md` with YAML frontmatter (date, source URL, tags).

The integration is silent: if the config file is absent or `enabled = false`, nothing changes.
```

**Step 3: Commit**

```bash
git add config.example.toml README.md
git commit -m "docs: add Obsidian integration config example and README section"
```

---

### Task 5: Smoke test end-to-end (manual)

**Goal:** Verify the full path works on your machine before declaring done.

**Step 1: Create your personal config**

```bash
mkdir -p ~/.config/yt-llm
cat > ~/.config/yt-llm/config.toml << 'EOF'
[obsidian]
enabled = true
vault_path = "~/Documents/obsidian"
inbox_dir = "Inbox"
tags = ["video-notes", "auto-generated"]
EOF
```

**Step 2: Run a transcription with notes**

```bash
transcribe "https://www.youtube.com/watch?v=f8cfH5XX-XU" --format structured
```

**Step 3: Verify the file appeared in Obsidian inbox**

```bash
ls ~/Documents/obsidian/Inbox/*.md
```
Expected: a `.md` file named after the inferred title.

```bash
head -20 ~/Documents/obsidian/Inbox/<filename>.md
```
Expected: YAML frontmatter with `date:`, `source:`, `tags:` followed by the notes content.

**Step 4: Check service logs confirm Obsidian write**

```bash
docker logs yt-llm-service --tail 20
```
Expected line: `Obsidian note saved: /home/.../Documents/obsidian/Inbox/<title>.md`

---

## Summary

| Task | Files Created/Modified | Tests |
|------|------------------------|-------|
| 1 | `src/user_config.py` | `tests/test_user_config.py` (4 tests) |
| 2 | `src/obsidian_service.py` | `tests/test_obsidian_service.py` (7 tests) |
| 3 | `src/run_llm_api.py` | Covered by existing test suite |
| 4 | `config.example.toml`, `README.md` | N/A |
| 5 | — | Manual smoke test |
