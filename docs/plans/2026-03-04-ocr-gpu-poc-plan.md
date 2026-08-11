# OCR GPU POC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Docker-based CLI benchmark comparing PaddleOCR vs Surya OCR on GPU 1 (RTX 3090 Ti, 24GB) for video frame text extraction.

**Architecture:** Single Docker container with both OCR engines installed, pinned to GPU 1. A Python CLI script (`benchmark.py`) processes test frame PNGs through both engines sequentially, measuring speed, VRAM, and text output. Results are JSON.

**Tech Stack:** PyTorch 2.x + CUDA 12.2, PaddleOCR (paddlepaddle-gpu + paddleocr), Surya OCR (surya-ocr), pynvml for VRAM measurement.

---

### Task 1: Create ocr-poc directory and requirements.txt

**Files:**
- Create: `ocr-poc/requirements.txt`

**Step 1: Create directory and requirements file**

```
# ocr-poc/requirements.txt
paddlepaddle-gpu>=3.0.0
paddleocr>=2.7.0
surya-ocr>=0.6.0
Pillow>=10.0.0
pynvml>=12.0.0
```

**Step 2: Commit**

```bash
git add ocr-poc/requirements.txt
git commit -m "chore: add ocr-poc requirements (PaddleOCR + Surya)"
```

---

### Task 2: Create Dockerfile

**Files:**
- Create: `ocr-poc/Dockerfile`

**Step 1: Write Dockerfile**

Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (closest match for driver 590.48, CUDA 12.x).

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps for PaddleOCR (libGL, libglib for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY benchmark.py .

ENTRYPOINT ["python", "benchmark.py"]
```

**Step 2: Commit**

```bash
git add ocr-poc/Dockerfile
git commit -m "chore: add ocr-poc Dockerfile (PyTorch CUDA + PaddleOCR + Surya)"
```

---

### Task 3: Create docker-compose.yml

**Files:**
- Create: `ocr-poc/docker-compose.yml`

**Step 1: Write compose file pinned to GPU 1**

```yaml
services:
  ocr-poc:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ocr-poc
    volumes:
      - ./test_frames:/app/test_frames:ro
      - ./results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0  # maps to physical GPU 1 via device_ids below
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

Note: `device_ids: ['1']` exposes physical GPU 1 as cuda:0 inside the container. `CUDA_VISIBLE_DEVICES=0` targets that single visible device.

**Step 2: Commit**

```bash
git add ocr-poc/docker-compose.yml
git commit -m "chore: add ocr-poc docker-compose (GPU 1)"
```

---

### Task 4: Create test_frames directory with README

**Files:**
- Create: `ocr-poc/test_frames/README.md`

**Step 1: Write instructions for populating test frames**

```markdown
# Test Frames

Place 10-20 representative PNG frames here for benchmarking.

## Recommended frame types:
1. Terminal/shell output (dark background, light text)
2. Code editor (VS Code, vim) with syntax highlighting
3. Presentation slides with bullet points
4. Browser content with navigation chrome
5. Mixed content (code + diagrams + text)

## How to extract frames from a YouTube video:

```bash
# Single frame at timestamp 5:30
ffmpeg -ss 5:30 -i video.mp4 -frames:v 1 terminal_dark.png

# Range of frames at 1fps
ffmpeg -ss 8:29 -to 9:15 -i video.mp4 -vf fps=1 slide_%04d.png
```

## Naming convention:
- `terminal_dark_01.png` — dark terminal
- `code_vscode_01.png` — VS Code editor
- `slide_text_01.png` — presentation slide
- `browser_article_01.png` — browser content
```

**Step 2: Commit**

```bash
git add ocr-poc/test_frames/README.md
git commit -m "docs: add test_frames README with extraction instructions"
```

---

### Task 5: Write benchmark.py — argument parsing and main structure

**Files:**
- Create: `ocr-poc/benchmark.py`

**Step 1: Write the CLI skeleton with argparse**

```python
#!/usr/bin/env python3
"""Benchmark PaddleOCR vs Surya OCR on video frame text extraction.

Usage:
    python benchmark.py --frames-dir test_frames/ --output results/report.json
    python benchmark.py --frames-dir test_frames/ --engine paddle  # single engine
    python benchmark.py --frames-dir test_frames/ --engine surya   # single engine
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pynvml
from PIL import Image


@dataclass
class FrameResult:
    frame: str
    engine: str
    text: str
    confidence: float
    time_ms: float
    vram_before_mb: float
    vram_after_mb: float


@dataclass
class EngineSummary:
    engine: str
    avg_time_ms: float
    median_time_ms: float
    peak_vram_mb: float
    total_frames: int
    failed_frames: int


@dataclass
class BenchmarkReport:
    gpu_name: str
    vram_total_mb: int
    frames_dir: str
    frame_count: int
    results: list = field(default_factory=list)
    summaries: list = field(default_factory=list)


def get_gpu_info() -> tuple[str, int]:
    """Return (gpu_name, vram_total_mb) for the current CUDA device."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return name, mem.total // (1024 * 1024)


def get_vram_used_mb() -> float:
    """Return current VRAM usage in MB."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / (1024 * 1024)


def load_frames(frames_dir: str) -> list[tuple[str, Image.Image]]:
    """Load all PNG/JPG frames from directory. Returns (filename, PIL.Image) pairs."""
    frames = []
    for fname in sorted(os.listdir(frames_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(frames_dir, fname)
            img = Image.open(path).convert("RGB")
            frames.append((fname, img))
    return frames


def run_paddle(frames: list[tuple[str, Image.Image]]) -> list[FrameResult]:
    """Run PaddleOCR on all frames. Returns list of FrameResult."""
    import numpy as np
    from paddleocr import PaddleOCR

    print("[paddle] Initializing PaddleOCR...", file=sys.stderr)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    results = []
    for fname, img in frames:
        img_np = np.array(img)
        vram_before = get_vram_used_mb()

        start = time.perf_counter()
        try:
            raw = ocr.ocr(img_np, cls=True)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract text and confidence
            lines = []
            confidences = []
            if raw and raw[0]:
                for line in raw[0]:
                    text, conf = line[1]
                    lines.append(text)
                    confidences.append(conf)

            text = "\n".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            text = f"ERROR: {e}"
            avg_conf = 0.0

        vram_after = get_vram_used_mb()
        results.append(FrameResult(
            frame=fname, engine="paddle", text=text,
            confidence=avg_conf, time_ms=elapsed_ms,
            vram_before_mb=vram_before, vram_after_mb=vram_after,
        ))
        print(f"  [paddle] {fname}: {elapsed_ms:.0f}ms, {len(text)} chars", file=sys.stderr)

    return results


def run_surya(frames: list[tuple[str, Image.Image]]) -> list[FrameResult]:
    """Run Surya OCR on all frames. Returns list of FrameResult."""
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    print("[surya] Initializing Surya OCR...", file=sys.stderr)
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    results = []
    for fname, img in frames:
        vram_before = get_vram_used_mb()

        start = time.perf_counter()
        try:
            predictions = rec_predictor([img], det_predictor=det_predictor)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract text from predictions
            lines = []
            confidences = []
            if predictions and len(predictions) > 0:
                for text_line in predictions[0].text_lines:
                    lines.append(text_line.text)
                    confidences.append(text_line.confidence)

            text = "\n".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            text = f"ERROR: {e}"
            avg_conf = 0.0

        vram_after = get_vram_used_mb()
        results.append(FrameResult(
            frame=fname, engine="surya", text=text,
            confidence=avg_conf, time_ms=elapsed_ms,
            vram_before_mb=vram_before, vram_after_mb=vram_after,
        ))
        print(f"  [surya] {fname}: {elapsed_ms:.0f}ms, {len(text)} chars", file=sys.stderr)

    return results


def compute_summary(results: list[FrameResult], engine: str) -> EngineSummary:
    """Compute summary stats for one engine's results."""
    times = [r.time_ms for r in results if not r.text.startswith("ERROR")]
    peak_vram = max(r.vram_after_mb for r in results) if results else 0
    failed = sum(1 for r in results if r.text.startswith("ERROR"))

    if times:
        avg_t = sum(times) / len(times)
        sorted_t = sorted(times)
        median_t = sorted_t[len(sorted_t) // 2]
    else:
        avg_t = median_t = 0.0

    return EngineSummary(
        engine=engine, avg_time_ms=avg_t, median_time_ms=median_t,
        peak_vram_mb=peak_vram, total_frames=len(results), failed_frames=failed,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR vs Surya OCR")
    parser.add_argument("--frames-dir", default="test_frames", help="Directory with PNG frames")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--engine", choices=["paddle", "surya", "both"], default="both",
                        help="Which engine(s) to benchmark")
    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"Error: {args.frames_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pynvml.nvmlInit()
    gpu_name, vram_total = get_gpu_info()
    print(f"GPU: {gpu_name} ({vram_total} MB)", file=sys.stderr)

    frames = load_frames(args.frames_dir)
    if not frames:
        print(f"Error: no PNG/JPG frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(frames)} frames", file=sys.stderr)

    report = BenchmarkReport(
        gpu_name=gpu_name, vram_total_mb=vram_total,
        frames_dir=args.frames_dir, frame_count=len(frames),
    )

    if args.engine in ("paddle", "both"):
        paddle_results = run_paddle(frames)
        report.results.extend(paddle_results)
        report.summaries.append(asdict(compute_summary(paddle_results, "paddle")))

    if args.engine in ("surya", "both"):
        surya_results = run_surya(frames)
        report.results.extend(surya_results)
        report.summaries.append(asdict(compute_summary(surya_results, "surya")))

    # Serialize
    output_data = asdict(report)
    output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Print summary table to stderr
    print("\n=== SUMMARY ===", file=sys.stderr)
    for s in report.summaries:
        print(f"  {s['engine']:>8s}: avg={s['avg_time_ms']:.0f}ms  "
              f"median={s['median_time_ms']:.0f}ms  "
              f"peak_vram={s['peak_vram_mb']:.0f}MB  "
              f"failed={s['failed_frames']}/{s['total_frames']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add ocr-poc/benchmark.py
git commit -m "feat: add OCR benchmark script (PaddleOCR vs Surya)"
```

---

### Task 6: Create start.sh convenience script

**Files:**
- Create: `ocr-poc/start.sh`

**Step 1: Write start script**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Build if needed
docker compose build

# Create results directory
mkdir -p results

# Run benchmark
docker compose run --rm ocr-poc \
    --frames-dir /app/test_frames \
    --output /app/results/report.json \
    "$@"

echo ""
echo "Results saved to: ocr-poc/results/report.json"
```

**Step 2: Make executable and commit**

```bash
chmod +x ocr-poc/start.sh
git add ocr-poc/start.sh
git commit -m "chore: add start.sh for OCR POC benchmark"
```

---

### Task 7: Populate test frames and run first benchmark

**Step 1: Extract test frames from a real YouTube video**

Use ffmpeg to capture representative frames. Pick a coding tutorial or tech talk video that has terminal output, slides, and code editor views.

```bash
cd ocr-poc/test_frames

# Example: extract 1 frame per second from a 30-second range
# Replace VIDEO_PATH with an actual downloaded video
ffmpeg -ss 5:00 -to 5:30 -i VIDEO_PATH -vf fps=1 slide_%04d.png

# Or manually screenshot terminal, VS Code, slides and save as PNG
```

Need at minimum 5 frames to run a meaningful benchmark.

**Step 2: Run the benchmark**

```bash
cd ocr-poc
./start.sh
```

Expected: JSON report in `ocr-poc/results/report.json` with per-frame timings, VRAM usage, and text output for both engines.

**Step 3: Review results**

```bash
cat results/report.json | python3 -m json.tool | head -60
```

Check:
- Both engines produced text (not ERROR)
- Speed comparison (avg_time_ms)
- VRAM usage (peak_vram_mb)
- Text quality (visual inspection)

---

## File Summary

| File | Purpose |
|------|---------|
| `ocr-poc/requirements.txt` | PaddleOCR + Surya + pynvml deps |
| `ocr-poc/Dockerfile` | PyTorch CUDA container with both engines |
| `ocr-poc/docker-compose.yml` | GPU 1 assignment, volume mounts |
| `ocr-poc/benchmark.py` | CLI benchmark script |
| `ocr-poc/start.sh` | One-command build + run |
| `ocr-poc/test_frames/README.md` | Instructions for populating test data |
| `ocr-poc/results/` | Output directory (gitignored) |
