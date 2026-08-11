"""Test session setup.

Everything the suite needs to import `src/` modules lives here rather than in
the test command, so `pytest tests/` works from a bare checkout and `make test`
stays a single line.

Three things have to be true *before* any test module is imported, because
`run_llm_api.py` builds a `Config()` and an `AudioDownloader()` at import time:

1. `tests/stubs`, the repo root, and `src/` are on `sys.path`.
2. `TEMP_DIR` / `OUTPUT_DIR` point somewhere writable — `Config.__init__`
   mkdirs them, and the defaults (`/app/tmp`, `/app/output`) only exist inside
   the container.
3. A `yt-dlp` binary is on `PATH` — `AudioDownloader._ensure_yt_dlp()` raises
   without one. Tests mock every actual download, so a no-op shim is enough.
"""

import atexit
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

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
