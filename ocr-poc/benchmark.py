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
