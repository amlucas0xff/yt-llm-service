# Session Notes

## Tests and smoke

Run `make test` for the mocked suite; `conftest.py` prepares writable directories, a `yt-dlp` shim, and import paths before collection. `run_llm_api.py` creates `Config()` and `AudioDownloader()` at import time, so fixture-only setup is too late. See README.md's "Running tests" for details.

Run `make smoke` for real WhisperX and llama-cpp coverage. It uses a disposable API, output directory, and Obsidian vault while reusing the Compose sidecar; it leaves live output and notes untouched. A green mocked suite alone does not prove the real pipeline works.

Historical OCR proof-of-concept decision: `docs/adr/0002-abandonar-ocr-de-video.md`.
