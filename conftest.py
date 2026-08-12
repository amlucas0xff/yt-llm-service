"""Test session setup.

Everything the suite needs to import `src/` modules lives here rather than in
the test command, so `pytest tests/` works from a bare checkout and `make test`
stays a single line.

Four things have to be true *before* any test module is imported, because
`run_llm_api.py` builds a `Config()` and an `AudioDownloader()` at import time:

1. `tests/stubs`, the repo root, and `src/` are on `sys.path`.
2. `TEMP_DIR` / `OUTPUT_DIR` point somewhere writable — `Config.__init__`
   mkdirs them, and the defaults (`/app/tmp`, `/app/output`) only exist inside
   the container.
3. A `yt-dlp` binary is on `PATH` — `AudioDownloader._ensure_yt_dlp()` raises
   without one. Tests mock every actual download, so a no-op shim is enough.
4. `python-dotenv` loads nothing. See below.

That "before any test module is imported" is why these are plain module-level
statements rather than autouse fixtures: fixtures run too late to affect an
import-time `Config()`.
"""

import atexit
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import dotenv
import pytest

_ROOT = Path(__file__).parent

# tests/stubs shadows heavyweight ML packages — see tests/stubs/torch.py.
sys.path.insert(0, str(_ROOT / "tests" / "stubs"))
sys.path.insert(1, str(_ROOT))
sys.path.insert(2, str(_ROOT / "src"))

_SANDBOX = Path(tempfile.mkdtemp(prefix="yt-llm-tests-"))
atexit.register(shutil.rmtree, _SANDBOX, True)

os.environ["TEMP_DIR"] = str(_SANDBOX / "tmp")
os.environ["OUTPUT_DIR"] = str(_SANDBOX / "output")

_shim_dir = _SANDBOX / "bin"
_shim_dir.mkdir(parents=True, exist_ok=True)
_shim = _shim_dir / "yt-dlp"
_shim.write_text("#!/bin/sh\nexit 0\n")
_shim.chmod(_shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
os.environ["PATH"] = f"{_shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"

# `Config.__init__` calls `load_dotenv()` with no arguments, which searches
# upwards from src/config.py and finds the repo-root `.env`. Left alone, every
# test that builds a Config inherits the developer's real HF_TOKEN,
# HOST_OUTPUT_DIR and the rest — so the suite's result depends on a gitignored
# file that differs on every machine, and green here means nothing about green
# anywhere else. Neutralise it once, for everyone, rather than per test.
_real_load_dotenv = dotenv.load_dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False


@pytest.fixture
def real_dotenv():
    """Opt back in to the genuine python-dotenv loader for one test.

    For tests *about* env loading. Anything else wanting real settings should
    set the variables it needs explicitly — that way the test states its own
    inputs instead of depending on the machine it runs on.
    """
    return _real_load_dotenv
