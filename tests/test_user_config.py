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


def test_load_handles_bad_value_type(tmp_path):
    from user_config import load_user_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[obsidian]\nenabled = true\nvault_path = 123\n')
    assert load_user_config(config_path=cfg_file) is None
