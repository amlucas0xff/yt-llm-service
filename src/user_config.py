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

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10 backport
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
        obsidian = data.get("obsidian", {})
        vault_raw = obsidian.get("vault_path", "~/Documents/obsidian")
        if not isinstance(vault_raw, str):
            raise TypeError(f"vault_path must be a string, got {type(vault_raw).__name__}")
        return UserConfig(
            obsidian_enabled=obsidian.get("enabled", False),
            vault_path=Path(vault_raw).expanduser(),
            inbox_dir=obsidian.get("inbox_dir", "Inbox"),
            tags=obsidian.get("tags", ["video-notes"]),
        )
    except Exception as e:
        logger.warning(f"Failed to load user config at {config_path}: {e}")
        return None
