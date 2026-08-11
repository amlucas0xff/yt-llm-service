"""Paths written into the vault must be openable from the host.

The service sees /app/output; the vault is read on the host, where that path
does not exist. A provenance link the reader cannot follow is not provenance.
"""
import os
from unittest.mock import patch

from config import Config


def make_config(tmp_path, host_output_dir=None):
    """Config with a writable OUTPUT_DIR — Config.__init__ mkdirs it.

    load_dotenv is stubbed out: Config() otherwise reads the developer's real
    .env from the repo root, so whatever HOST_OUTPUT_DIR they happen to have
    set would decide the result of these tests.
    """
    container_out = tmp_path / "container-output"
    env = {"OUTPUT_DIR": str(container_out), "TEMP_DIR": str(tmp_path / "tmp")}
    if host_output_dir:
        env["HOST_OUTPUT_DIR"] = host_output_dir
    with patch.dict(os.environ, env, clear=False), \
            patch("dotenv.load_dotenv", lambda *a, **k: None):
        if not host_output_dir:
            os.environ.pop("HOST_OUTPUT_DIR", None)
        cfg = Config()
    return cfg, str(container_out)


def test_translates_container_path_to_host_path(tmp_path):
    cfg, container_out = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path(f"{container_out}/Talk/transcription_1.md") == (
        "/home/me/vault/transcripts/Talk/transcription_1.md"
    )


def test_leaves_path_alone_when_no_host_mapping_is_configured(tmp_path):
    cfg, container_out = make_config(tmp_path)
    original = f"{container_out}/Talk/transcription_1.md"
    assert cfg.to_host_path(original) == original


def test_leaves_unrelated_paths_alone(tmp_path):
    cfg, _ = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path("/somewhere/else/file.md") == "/somewhere/else/file.md"


def test_handles_none(tmp_path):
    cfg, _ = make_config(tmp_path, host_output_dir="/home/me/vault/transcripts")
    assert cfg.to_host_path(None) is None
