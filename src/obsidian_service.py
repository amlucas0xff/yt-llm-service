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

    @staticmethod
    def _yaml_quote(value: str) -> str:
        """Single-quote a scalar so YAML survives it.

        Transcript paths sit under a user-configurable OUTPUT_DIR, so they can
        contain ': ' — which makes an unquoted value unparseable and silently
        costs the note all its frontmatter in Obsidian.
        """
        return "'" + value.replace("'", "''") + "'"

    def _build_frontmatter(
        self,
        source_url: Optional[str],
        source_transcript: Optional[str],
        truncated: bool,
    ) -> str:
        lines = [
            "---",
            f"date: {date.today().isoformat()}",
        ]
        if source_url:
            lines.append(f"source: {source_url}")
        if source_transcript:
            lines.append(f"source_transcript: {self._yaml_quote(source_transcript)}")
        lines.append(f"truncated: {'true' if truncated else 'false'}")
        if self.tags:
            lines.append("tags:")
            for tag in self.tags:
                lines.append(f"  - {tag}")
        lines.append("---")
        return "\n".join(lines) + "\n\n"

    def save_note(
        self,
        notes_text: str,
        source_url: Optional[str] = None,
        source_transcript: Optional[str] = None,
        truncated: bool = False,
    ) -> Optional[str]:
        """
        Write notes to {vault_path}/{inbox_dir}/{title}.md with YAML frontmatter.

        Args:
            notes_text: The generated markdown notes.
            source_url: YouTube URL the notes came from, when there is one.
            source_transcript: Path to the transcript markdown these notes
                summarize, so the note points back at its own evidence.
            truncated: Whether the transcript's middle was dropped before the
                model saw it — the notes then cover only its head and tail.

        Returns the absolute path string on success, None on failure.
        """
        try:
            inbox = self.vault_path / self.inbox_dir
            inbox.mkdir(parents=True, exist_ok=True)

            title = self._extract_title(notes_text)
            filename = self._sanitize_filename(title) + ".md"
            file_path = inbox / filename

            frontmatter = self._build_frontmatter(source_url, source_transcript, truncated)
            content = frontmatter + notes_text

            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Obsidian note saved: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save Obsidian note: {e}")
            return None
