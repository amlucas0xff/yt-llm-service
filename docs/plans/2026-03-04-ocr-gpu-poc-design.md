# OCR GPU POC Design — PaddleOCR vs Surya on RTX 3090 Ti

**Date:** 2026-03-04
**Status:** Approved
**Goal:** Benchmark PaddleOCR and Surya OCR on GPU 1 (RTX 3090 Ti, 24GB) to choose a GPU-accelerated OCR engine for the video frame text extraction pipeline.

## Context

The current OCR pipeline (feature/ocr-migration branch) uses CPU-only Tesseract inside the yt-llm-service container on GPU 0 (RTX 4060, 8GB). This is slow (230-1000ms per frame) and wastes the 3090 Ti which sits idle most of the time (llama-cpp auto-unloads after 5min).

## Decision: Dedicated Container on GPU 1

- **Option C from analysis:** New Docker container pinned to GPU 1
- Follows existing GPU-isolation pattern (yt-llm-service=GPU0, llama-cpp=GPU1)
- Shares GPU 1 with llama-cpp; no conflict since llama-cpp idles between requests
- 24GB VRAM removes all library constraints (even Surya's 24GB default fits)

## Architecture

```
ocr-poc/
  Dockerfile           # PyTorch CUDA base + PaddleOCR + Surya
  benchmark.py         # CLI benchmark script
  requirements.txt     # paddlepaddle-gpu, paddleocr, surya-ocr
  test_frames/         # 10-20 representative PNG frames
  docker-compose.yml   # Standalone, GPU 1 assignment
```

Single container with both libraries installed. CLI-only (no API).

## Benchmark Script

`benchmark.py` processes all frames in `test_frames/` through both engines and outputs:

```json
{
  "environment": { "gpu": "RTX 3090 Ti", "vram_total_mb": 24576 },
  "results": [
    {
      "frame": "terminal_dark.png",
      "paddle": { "text": "...", "time_ms": 95, "vram_mb": 2500, "confidence": 0.94 },
      "surya":  { "text": "...", "time_ms": 180, "vram_mb": 4200, "confidence": 0.97 }
    }
  ],
  "summary": {
    "paddle": { "avg_time_ms": 110, "peak_vram_mb": 2800 },
    "surya":  { "avg_time_ms": 200, "peak_vram_mb": 5100 }
  }
}
```

Metrics per frame: extracted text, inference time (ms), VRAM usage (MB), confidence score.

## Test Frames

Curated set of ~10-20 PNGs representing real use cases:
- Terminal/shell output (dark background)
- Code editor screenshots (VS Code, vim)
- Presentation slides with text
- Mixed content (code + diagrams)
- Browser content with UI chrome

Source: Extract frames from actual YouTube videos using the existing ffmpeg pipeline.

## GPU 1 Sharing Strategy

llama-cpp auto-unloads after 5min idle (`LLAMA_CPP_IDLE_SECONDS=300`). For manual POC benchmarking, simply avoid running OCR during active transcription+notes jobs. No orchestration needed.

## Success Criteria

1. Both engines produce readable text from all frame types
2. Benchmark report with speed/VRAM/accuracy comparison
3. Clear winner identified for production integration

## Non-Goals

- No API endpoint (CLI-only POC)
- No integration with yt-llm-service pipeline
- No production Dockerfile optimization
- No automated test frame generation
