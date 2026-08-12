# Session Notes

## 2026-03-02 verification environment

- What failed and why:
  Running tests inside the service image via `/app/.venv/bin/pytest` failed because the image's virtualenv entrypoints point at a non-existent host-specific Python path. Using the image's system Python without the venv also failed because CLI/API dependencies were not on `sys.path`.
- Working approach:
  Run `make test`. Superseded as of 2026-08-11: the manual recipe this note used to carry — `uv run` with hand-set `TEMP_DIR`/`OUTPUT_DIR`, a `yt-dlp` shim on `PATH`, and a three-entry `PYTHONPATH` — now lives in `conftest.py` (commit `05aede7`), so `pytest tests/` works from a bare checkout too. See "Running tests" in `README.md`.
- Foundations needed:
  `run_llm_api.py` instantiates `Config()` and `AudioDownloader()` at import time, so tests need writable app directories and a discoverable `yt-dlp` binary even when all runtime behavior is mocked. This is also why `conftest.py` sets everything up in module-level statements rather than fixtures — fixtures run after collection has already imported the module.
- Do not trust a green suite as proof the system works:
  `make test` is fully mocked and stayed green through a guaranteed yt-dlp crash, an unbuildable sidecar image, a completely dead Obsidian integration, and a broken provenance link. All four surfaced only from `./scripts/smoke.sh`, which is the only check that runs WhisperX and llama-cpp for real. Run it before believing the suite.

## 2026-03-05 OCR GPU POC dependency conflicts

- What failed and why:
  PaddlePaddle GPU and PyTorch both pin exact NVIDIA CUDA runtime versions (paddle: nvrtc 12.3.107, torch: nvrtc 12.4.127). pip cannot resolve both. Surya v0.17+ requires torch>=2.7 which conflicts with the pytorch:2.5.1 base image. Surya v0.6.13 works with torch 2.5 but needs transformers<4.46.
- Working approach:
  Install paddlepaddle-gpu with `--no-deps` from PaddlePaddle's own index (`https://www.paddlepaddle.org.cn/packages/stable/cu123/`), then install its Python-only deps (protobuf, numpy, decorator, opt-einsum) separately. Pin surya-ocr==0.6.13 with `--no-deps` and install its deps with transformers<4.46.
- Foundations needed:
  The PyTorch CUDA base image already has CUDA runtime libs, so `--no-deps` on paddle is safe (skips redundant NVIDIA wheels). PaddleOCR v3+ uses `predict()` not `ocr()`, and returns dict-like OCRResult with `rec_texts`/`rec_scores` keys. Surya v0.6 uses functional API (`run_ocr`) not class-based predictors.

## 2026-03-05 CLIP OCR POC setup/runtime pitfalls

- What failed and why:
  `uv` with `[tool.uv].extra-index-url` pointed all packages at the PyTorch CUDA index, which broke resolution for unrelated packages (for example `markupsafe`) on CPython 3.13. `openai-clip` also failed when paired with `setuptools>=70` because it imports `pkg_resources.packaging`.
- Working approach:
  Use `[[tool.uv.index]]` + `[tool.uv.sources]` so only `torch`/`torchvision` resolve from `https://download.pytorch.org/whl/cu124`. Pin `setuptools<70` in the POC env to keep `pkg_resources.packaging` available for `openai-clip`.
- Foundations needed:
  CLIP tile scores can cluster tightly around `0.5`; threshold defaults may not crop. For the screenshot `Captura de tela 2026-03-05 135912.png`, `--threshold 0.53` produced meaningful filtering (~73% OCR line reduction), while `0.54+` fell back to full-frame.
