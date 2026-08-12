"""The suite must not read anyone's .env file.

`Config.__init__` calls `load_dotenv()` with no arguments, which searches
upwards from `src/config.py` and finds the repo-root `.env`. Every test that
builds a `Config` therefore inherits whatever the developer happens to have
configured, and the suite passes or fails based on a file that is gitignored
and different on every machine. That is not a hypothetical: a real
`HOST_OUTPUT_DIR` once overrode the intent of tests/test_host_paths.py.

conftest.py neutralises python-dotenv for the whole session. These tests hold
it to that.
"""
import os

import dotenv

from config import Config


def test_load_dotenv_does_not_read_a_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("YT_LLM_ENV_LEAK_CANARY=leaked\n")

    dotenv.load_dotenv(env_file)

    assert "YT_LLM_ENV_LEAK_CANARY" not in os.environ


def test_config_does_not_absorb_an_env_file(tmp_path):
    """A canary, not HOST_OUTPUT_DIR.

    Asserting on a real setting hides the bug instead of catching it: the
    repo-root .env has already been read by the time this runs, so the key is
    in os.environ, and load_dotenv declines to overwrite what is already set.
    The assertion then passes while the leak it was written to catch is
    happening. A name nothing else can define has no such escape.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("YT_LLM_CONFIG_LEAK_CANARY=leaked\n")

    Config(env_file=str(env_file))

    assert "YT_LLM_CONFIG_LEAK_CANARY" not in os.environ


def test_real_dotenv_fixture_hands_back_the_genuine_loader(tmp_path, real_dotenv):
    """The escape hatch has to actually work, or nobody can test env loading."""
    env_file = tmp_path / ".env"
    env_file.write_text("YT_LLM_ENV_OPT_IN_CANARY=loaded\n")

    try:
        real_dotenv(env_file)
        assert os.environ.get("YT_LLM_ENV_OPT_IN_CANARY") == "loaded"
    finally:
        os.environ.pop("YT_LLM_ENV_OPT_IN_CANARY", None)
