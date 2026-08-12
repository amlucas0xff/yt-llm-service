"""Paths written into the vault must be openable from the host.

The service sees /app/output; the vault is read on the host, where that path
does not exist. A provenance link the reader cannot follow is not provenance.
"""
import os
from unittest.mock import patch

from config import Config


def make_config(tmp_path, host_output_dir=None):
    """Config with a writable OUTPUT_DIR — Config.__init__ mkdirs it."""
    container_out = tmp_path / "container-output"
    env = {"OUTPUT_DIR": str(container_out), "TEMP_DIR": str(tmp_path / "tmp")}
    if host_output_dir:
        env["HOST_OUTPUT_DIR"] = host_output_dir
    with patch.dict(os.environ, env, clear=False):
        if not host_output_dir:
            os.environ.pop("HOST_OUTPUT_DIR", None)
        cfg = Config()
    return cfg, str(container_out)


def test_translates_container_path_to_host_path(tmp_path):
    cfg, container_out = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path(f"{container_out}/Talk/transcription_1.md") == (
        "/home/me/vault/transcripts/Talk/transcription_1.md"
    )


def test_gives_up_when_no_host_mapping_is_configured(tmp_path):
    """None, not the container path.

    Handing back /app/output/... is the very bug this exists to fix: the
    caller writes it into vault frontmatter, where it looks like provenance
    and opens nothing.
    """
    cfg, container_out = make_config(tmp_path)
    assert cfg.to_host_path(f"{container_out}/Talk/transcription_1.md") is None


def test_leaves_unrelated_paths_alone(tmp_path):
    cfg, _ = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path("/somewhere/else/file.md") == "/somewhere/else/file.md"


def test_leaves_path_alone_when_the_host_mapping_is_relative(tmp_path):
    """docker-compose defaults HOST_OUTPUT_DIR to ./data/output.

    A relative path in the frontmatter is worse than no path at all: it looks
    like provenance, but it only resolves for someone standing in the repo,
    and the whole point is that a human opens the note from the vault. The
    docstring promises this never guesses — so leave it alone.
    """
    cfg, container_out = make_config(tmp_path, host_output_dir="./data/output")
    assert cfg.to_host_path(f"{container_out}/Talk/transcription_1.md") is None


def test_handles_none(tmp_path):
    cfg, _ = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path(None) is None
