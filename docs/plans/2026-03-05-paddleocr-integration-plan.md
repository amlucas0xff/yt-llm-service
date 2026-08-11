# PaddleOCR Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add standalone OCR endpoints (`/ocr-youtube`, `/ocr-file`) that extract text from video frames using PaddleOCR, deployed as a thin sidecar container on GPU 1.

**Architecture:** Thin sidecar pattern. A new `ocr-service` FastAPI container (port 8003, GPU 1) receives images and returns OCR text. The main `yt-llm-service` orchestrates: downloads video, extracts frames via ffmpeg scene detection, sends frames to the sidecar, and returns structured results.

**Tech Stack:** PaddleOCR v3 (PP-OCRv5), FastAPI, httpx (async client), ffmpeg (scene detection), pynvml, Docker Compose.

---

### Task 1: Create ocr-service Dockerfile and requirements.txt

**Files:**
- Create: `ocr-service/requirements.txt`
- Create: `ocr-service/Dockerfile`

**Step 1: Create requirements.txt**

```
paddlepaddle-gpu==3.1.0
paddleocr>=2.7.0
fastapi>=0.104.0
uvicorn>=0.23.0
pynvml>=12.0.0
python-multipart>=0.0.6
Pillow>=10.0.0
```

**Step 2: Create Dockerfile**

Based on the proven `ocr-poc/Dockerfile` pattern. Key: install `paddlepaddle-gpu` with `--no-deps` from PaddlePaddle's own index to avoid NVIDIA CUDA runtime conflicts with PyTorch.

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PaddlePaddle GPU with --no-deps to avoid CUDA runtime conflicts
RUN pip install --no-cache-dir --no-deps \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
    paddlepaddle-gpu==3.1.0 && \
    pip install --no-cache-dir protobuf numpy decorator opt-einsum

# PaddleOCR + remaining deps
RUN pip install --no-cache-dir paddleocr>=2.7.0

# FastAPI + web deps
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.23.0 \
    pynvml>=12.0.0 \
    python-multipart>=0.0.6 \
    Pillow>=10.0.0

COPY app.py .

EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
```

**Step 3: Commit**

```bash
git add ocr-service/requirements.txt ocr-service/Dockerfile
git commit -m "chore: add ocr-service Dockerfile and requirements"
```

---

### Task 2: Create ocr-service FastAPI app

**Files:**
- Create: `ocr-service/app.py`

**Step 1: Write the sidecar FastAPI app**

This is the "dumb" sidecar: accepts images via multipart, runs PaddleOCR, returns text+confidence. PaddleOCR is initialized once at startup (lazy singleton to avoid model download on import).

```python
#!/usr/bin/env python3
"""OCR sidecar service — accepts images, returns extracted text via PaddleOCR."""

import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pynvml
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel


class ImageOCRResult(BaseModel):
    index: int
    texts: list[str]
    confidences: list[float]
    full_text: str
    avg_confidence: float


class OCRBatchResponse(BaseModel):
    results: list[ImageOCRResult]
    engine: str = "paddleocr"
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    engine: str
    gpu: str
    vram_used_mb: float
    vram_total_mb: float


# Global singleton — initialized on first request
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
    return _ocr_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    pynvml.nvmlInit()
    yield
    pynvml.nvmlShutdown()


app = FastAPI(title="OCR Service", lifespan=lifespan)


@app.post("/ocr", response_model=OCRBatchResponse)
async def ocr_batch(
    images: list[UploadFile] = File(...),
    lang: str = Form("en"),
):
    ocr = get_ocr_engine()
    start = time.perf_counter()
    results = []

    for idx, upload in enumerate(images):
        raw_bytes = await upload.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_np = np.array(img)

        texts = []
        confidences = []
        try:
            for result in ocr.predict(img_np):
                rec_texts = result.get("rec_texts", [])
                rec_scores = result.get("rec_scores", [])
                if rec_texts:
                    texts.extend(rec_texts)
                    confidences.extend(rec_scores)
        except Exception:
            pass  # skip failed frame, return empty

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        results.append(ImageOCRResult(
            index=idx,
            texts=texts,
            confidences=confidences,
            full_text="\n".join(texts),
            avg_confidence=avg_conf,
        ))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return OCRBatchResponse(results=results, processing_time_ms=elapsed_ms)


@app.get("/health", response_model=HealthResponse)
async def health():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return HealthResponse(
        status="ok",
        engine="paddleocr",
        gpu=name,
        vram_used_mb=mem.used / (1024 * 1024),
        vram_total_mb=mem.total / (1024 * 1024),
    )
```

**Step 2: Commit**

```bash
git add ocr-service/app.py
git commit -m "feat: add ocr-service FastAPI sidecar (PaddleOCR)"
```

---

### Task 3: Add ocr-service to docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Add ocr-service block**

Add after the `llama-cpp` service block. Pattern matches existing services: GPU isolation via `device_ids`, healthcheck, port mapping.

```yaml
  ocr-service:
    build: ./ocr-service
    container_name: ocr-service
    ports:
      - "8003:8003"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

**Step 2: Build and verify health endpoint**

```bash
docker compose build ocr-service
docker compose up -d ocr-service
curl -s http://localhost:8003/health | python3 -m json.tool
```

Expected: `{"status": "ok", "engine": "paddleocr", "gpu": "NVIDIA GeForce RTX 3090 Ti", ...}`

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add ocr-service to docker-compose (GPU 1)"
```

---

### Task 4: Create frame_extractor.py

**Files:**
- Create: `src/frame_extractor.py`
- Create: `tests/test_frame_extractor.py`

**Step 1: Write the frame extractor module**

Uses ffmpeg scene change detection. Parses stderr `showinfo` output for PTS timestamps. Returns list of `(frame_path, timestamp_s)` tuples.

```python
"""Extract key frames from video using ffmpeg scene change detection."""

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    path: str
    timestamp_s: float
    index: int


def extract_scene_frames(
    video_path: str,
    scene_threshold: float = 0.3,
    max_frames: int = 100,
) -> list[ExtractedFrame]:
    """Extract frames at scene changes from a video file.

    Uses ffmpeg select filter with scene change detection.
    Returns list of ExtractedFrame sorted by timestamp.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = tempfile.mkdtemp(prefix="ocr_frames_")
    output_pattern = str(Path(output_dir) / "frame_%04d.png")

    # ffmpeg scene detection with showinfo for timestamps
    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", f"select='gt(scene\\,{scene_threshold})',showinfo",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        output_pattern,
        "-y",
    ]

    logger.info(f"Extracting scene frames: threshold={scene_threshold}, max={max_frames}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )

    # Parse showinfo output from stderr for PTS timestamps
    # Pattern: [Parsed_showinfo_1 ...] pts_time:123.456
    pts_pattern = re.compile(r"pts_time:\s*([\d.]+)")
    timestamps = pts_pattern.findall(result.stderr)

    # Collect extracted frames
    frames = []
    frame_dir = Path(output_dir)
    for idx, frame_path in enumerate(sorted(frame_dir.glob("frame_*.png"))):
        ts = float(timestamps[idx]) if idx < len(timestamps) else 0.0
        frames.append(ExtractedFrame(
            path=str(frame_path),
            timestamp_s=ts,
            index=idx,
        ))

    logger.info(f"Extracted {len(frames)} scene-change frames")
    return frames


def cleanup_frames(frames: list[ExtractedFrame]) -> None:
    """Delete extracted frame files and their parent temp directory."""
    if not frames:
        return
    parent = Path(frames[0].path).parent
    for f in frames:
        Path(f.path).unlink(missing_ok=True)
    parent.rmdir()
```

**Step 2: Write tests**

```python
"""Tests for frame_extractor module."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from frame_extractor import extract_scene_frames, cleanup_frames, ExtractedFrame


def test_extract_scene_frames_calls_ffmpeg(tmp_path):
    """Verify ffmpeg is called with correct scene detection filter."""
    video = tmp_path / "test.mp4"
    video.touch()

    fake_stderr = (
        "[Parsed_showinfo_1 ...] pts_time:1.500\n"
        "[Parsed_showinfo_1 ...] pts_time:5.200\n"
    )

    # Create fake frame files that ffmpeg would produce
    with patch("frame_extractor.tempfile.mkdtemp") as mock_mkdtemp:
        mock_mkdtemp.return_value = str(tmp_path / "frames")
        (tmp_path / "frames").mkdir()
        (tmp_path / "frames" / "frame_0001.png").touch()
        (tmp_path / "frames" / "frame_0002.png").touch()

        with patch("frame_extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stderr=fake_stderr, returncode=0
            )
            frames = extract_scene_frames(str(video), scene_threshold=0.3, max_frames=50)

    assert len(frames) == 2
    assert frames[0].timestamp_s == 1.5
    assert frames[1].timestamp_s == 5.2
    assert frames[0].index == 0
    assert frames[1].index == 1

    # Verify ffmpeg was called with scene filter
    call_args = mock_run.call_args[0][0]
    assert "ffmpeg" in call_args[0]
    assert "select='gt(scene\\,0.3)'" in " ".join(call_args)


def test_extract_scene_frames_file_not_found():
    """Raise FileNotFoundError for missing video."""
    with pytest.raises(FileNotFoundError):
        extract_scene_frames("/nonexistent/video.mp4")


def test_cleanup_frames(tmp_path):
    """Verify cleanup deletes frames and parent dir."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    f1 = frame_dir / "frame_0001.png"
    f2 = frame_dir / "frame_0002.png"
    f1.touch()
    f2.touch()

    frames = [
        ExtractedFrame(path=str(f1), timestamp_s=1.0, index=0),
        ExtractedFrame(path=str(f2), timestamp_s=2.0, index=1),
    ]
    cleanup_frames(frames)

    assert not f1.exists()
    assert not f2.exists()
    assert not frame_dir.exists()


def test_cleanup_frames_empty_list():
    """cleanup_frames with empty list should not raise."""
    cleanup_frames([])
```

**Step 3: Run tests**

```bash
cd /home/amlucas/dev/yt-llm-service
PYTHONPATH=src:tests/stubs:. uv run pytest tests/test_frame_extractor.py -v
```

Expected: All 4 tests PASS.

**Step 4: Commit**

```bash
git add src/frame_extractor.py tests/test_frame_extractor.py
git commit -m "feat: add frame_extractor with ffmpeg scene detection"
```

---

### Task 5: Create ocr_client.py

**Files:**
- Create: `src/ocr_client.py`
- Create: `tests/test_ocr_client.py`

**Step 1: Write the async OCR client**

Follows the same httpx async pattern as `notes_service.py`. Sends images as multipart to the sidecar, returns structured results.

```python
"""Async HTTP client for the OCR sidecar service."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OCRFrameResult:
    index: int
    texts: list[str]
    confidences: list[float]
    full_text: str
    avg_confidence: float


@dataclass
class OCRBatchResult:
    results: list[OCRFrameResult]
    engine: str
    processing_time_ms: float


class OCRClient:
    """Async client for the ocr-service sidecar."""

    def __init__(self, base_url: str = "http://ocr-service:8003", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)

    async def ocr_images(
        self,
        image_paths: list[str],
        lang: str = "en",
    ) -> OCRBatchResult | None:
        """Send images to OCR service and return results.

        Returns None if the service is unavailable or errors out.
        """
        if not image_paths:
            return OCRBatchResult(results=[], engine="paddleocr", processing_time_ms=0.0)

        files = []
        for path in image_paths:
            p = Path(path)
            files.append(("images", (p.name, open(p, "rb"), "image/png")))

        url = f"{self.base_url}/ocr"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, files=files, data={"lang": lang})
                response.raise_for_status()

            data = response.json()
            results = [
                OCRFrameResult(
                    index=r["index"],
                    texts=r["texts"],
                    confidences=r["confidences"],
                    full_text=r["full_text"],
                    avg_confidence=r["avg_confidence"],
                )
                for r in data.get("results", [])
            ]
            return OCRBatchResult(
                results=results,
                engine=data.get("engine", "paddleocr"),
                processing_time_ms=data.get("processing_time_ms", 0.0),
            )

        except httpx.ConnectError as e:
            logger.warning(f"ocr-service unavailable: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.warning(f"ocr-service request timed out: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"ocr-service HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling ocr-service: {e}")
            return None
        finally:
            for _, (_, fobj, _) in files:
                fobj.close()

    async def health(self) -> dict | None:
        """Check ocr-service health. Returns health dict or None."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None
```

**Step 2: Write tests**

```python
"""Tests for ocr_client module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from ocr_client import OCRClient, OCRBatchResult


@pytest.mark.asyncio
async def test_ocr_images_sends_multipart(tmp_path):
    """Verify images are sent as multipart to the sidecar."""
    # Create fake image files
    img1 = tmp_path / "frame_0001.png"
    img2 = tmp_path / "frame_0002.png"
    img1.write_bytes(b"fake png 1")
    img2.write_bytes(b"fake png 2")

    client = OCRClient(base_url="http://localhost:8003")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "texts": ["hello"], "confidences": [0.99],
             "full_text": "hello", "avg_confidence": 0.99},
            {"index": 1, "texts": ["world"], "confidences": [0.95],
             "full_text": "world", "avg_confidence": 0.95},
        ],
        "engine": "paddleocr",
        "processing_time_ms": 500.0,
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_post = AsyncMock(return_value=mock_response)
        MockClient.return_value.__aenter__.return_value.post = mock_post
        result = await client.ocr_images([str(img1), str(img2)])

    assert result is not None
    assert len(result.results) == 2
    assert result.results[0].full_text == "hello"
    assert result.results[1].full_text == "world"
    assert result.engine == "paddleocr"


@pytest.mark.asyncio
async def test_ocr_images_empty_list():
    """Empty image list returns empty results, no HTTP call."""
    client = OCRClient()
    result = await client.ocr_images([])
    assert result is not None
    assert result.results == []


@pytest.mark.asyncio
async def test_ocr_images_service_unavailable():
    """Returns None when ocr-service is down."""
    client = OCRClient(base_url="http://localhost:8003")

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("connection refused")
        )
        result = await client.ocr_images(["/fake/img.png"])

    assert result is None


@pytest.mark.asyncio
async def test_health_returns_dict():
    """Health check returns parsed JSON."""
    client = OCRClient()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "ok", "engine": "paddleocr"}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        result = await client.health()

    assert result == {"status": "ok", "engine": "paddleocr"}
```

**Step 3: Run tests**

```bash
PYTHONPATH=src:tests/stubs:. uv run pytest tests/test_ocr_client.py -v
```

Expected: All 4 tests PASS.

**Step 4: Commit**

```bash
git add src/ocr_client.py tests/test_ocr_client.py
git commit -m "feat: add async OCR client for sidecar communication"
```

---

### Task 6: Add OCR config to Config class

**Files:**
- Modify: `src/config.py`

**Step 1: Add OCR env vars**

Add after the existing llama-cpp config block (around line 40):

```python
# OCR service configuration
self.OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr-service:8003")
self.OCR_SERVICE_TIMEOUT = float(os.getenv("OCR_SERVICE_TIMEOUT", "60"))
self.OCR_SCENE_THRESHOLD = float(os.getenv("OCR_SCENE_THRESHOLD", "0.3"))
self.OCR_MAX_FRAMES = int(os.getenv("OCR_MAX_FRAMES", "100"))
```

**Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: add OCR service config (URL, timeout, scene threshold, max frames)"
```

---

### Task 7: Create ocr_dedup.py — post-OCR text deduplication

> **Codex review finding:** This task was originally Task 8, after the endpoints task. Swapped because the endpoints import `ocr_dedup` at module load time — the module must exist first.

**Files:**
- Create: `src/ocr_dedup.py`
- Create: `tests/test_ocr_dedup.py`

**Context:** Consecutive video frames often contain overlapping text (persistent titles, watermarks, lower thirds). Scene detection reduces frame count but does not eliminate text overlap. This module merges OCR results from consecutive frames when their text is similar, producing time-spanning entries instead of per-frame duplicates.

**Algorithm:** Line-level set similarity. Split each frame's `full_text` into normalized lines, compute Jaccard similarity (intersection / union) between consecutive frames. If Jaccard >= threshold AND time gap <= `max_gap_seconds`, merge into a single span. This avoids the whole-text `SequenceMatcher` problem where "Chapter 1" and "Chapter 2" get a misleadingly high ratio (~0.89).

> **Why line-level, not whole-text?** `SequenceMatcher("chapter 1: introduction", "chapter 2: data structures").ratio()` ≈ 0.55 (correct split), but for short single-line texts like "Chapter 1" vs "Chapter 2" the ratio is ~0.89 (false merge). Line-level Jaccard treats each line as an atomic unit — "Chapter 1" != "Chapter 2" → Jaccard = 0.0 → correct split. For multi-line frames with shared persistent text (title bar) plus changing content, Jaccard captures the partial overlap naturally.

**References:**
- [ADNVideo text tracking](https://github.com/ADNVideo/ocr-processing/wiki/Text-boxes-tracking) — Levenshtein + geometric bbox matching, merged time ranges
- [video-text-extraction](https://github.com/Akashkalakonda/video-text-extraction) — SSIM frame-level dedup (pre-OCR)
- [FrameTextExtractor](https://github.com/zeynelacikgoez/FrameTextExtractor) — motion detection (pre-OCR)

**Step 1: Write the dedup module**

```python
"""Post-OCR deduplication — merge consecutive frames with overlapping text.

Uses line-level Jaccard similarity to avoid false merges on short texts
(e.g., "Chapter 1" vs "Chapter 2" would get ~0.89 with SequenceMatcher
but 0.0 with line-level Jaccard since the lines differ as atomic units).
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_MAX_GAP_SECONDS = 30.0


def _normalize_line(line: str) -> str:
    """Normalize a single line: lowercase, collapse whitespace, strip."""
    return " ".join(line.lower().split())


def _to_line_set(text: str) -> set[str]:
    """Split text into normalized non-empty lines."""
    return {
        _normalize_line(line)
        for line in text.split("\n")
        if line.strip()
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_ocr_results(
    entries: list[tuple[float, str, float]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> list[dict]:
    """Merge consecutive OCR entries with overlapping text into spans.

    Args:
        entries: List of (timestamp_s, full_text, avg_confidence) sorted by
                 timestamp. Empty-text entries are skipped.
        threshold: Line-level Jaccard similarity at or above which two
                   entries are merged.
        max_gap_seconds: Maximum time gap (seconds) between frames to allow
                         merging. Prevents merging distant frames that happen
                         to have similar text (e.g., recurring watermark after
                         a long gap).

    Returns:
        List of dicts with keys: start_time, end_time, text, confidence,
        frame_count. Sorted by start_time.
    """
    if not entries:
        return []

    # Filter out empty-text entries
    entries = [(ts, text, conf) for ts, text, conf in entries if text.strip()]
    if not entries:
        return []

    spans: list[dict] = []
    ts, text, conf = entries[0]
    current = {
        "start_time": ts,
        "end_time": ts,
        "text": text,
        "confidence": conf,
        "frame_count": 1,
        "_lines": _to_line_set(text),
    }

    for ts, text, conf in entries[1:]:
        lines = _to_line_set(text)
        sim = _jaccard(current["_lines"], lines)
        gap = ts - current["end_time"]

        if sim >= threshold and gap <= max_gap_seconds:
            # Merge: extend time range, keep higher-confidence text
            current["end_time"] = ts
            current["frame_count"] += 1
            if conf > current["confidence"]:
                current["text"] = text
                current["confidence"] = conf
                current["_lines"] = lines
        else:
            # New span
            spans.append(current)
            current = {
                "start_time": ts,
                "end_time": ts,
                "text": text,
                "confidence": conf,
                "frame_count": 1,
                "_lines": lines,
            }

    spans.append(current)

    # Remove internal key and return
    for s in spans:
        s.pop("_lines", None)

    logger.info(
        f"Dedup: {len(entries)} frames -> {len(spans)} spans "
        f"(threshold={threshold}, max_gap={max_gap_seconds}s)"
    )
    return spans
```

**Step 2: Write tests**

```python
"""Tests for ocr_dedup module."""

import pytest
from ocr_dedup import deduplicate_ocr_results, _jaccard, _to_line_set, _normalize_line


def test_empty_input():
    """Empty list returns empty list."""
    assert deduplicate_ocr_results([]) == []


def test_single_entry():
    """Single entry becomes a single span."""
    result = deduplicate_ocr_results([(1.0, "hello world", 0.95)])
    assert len(result) == 1
    assert result[0]["start_time"] == 1.0
    assert result[0]["end_time"] == 1.0
    assert result[0]["text"] == "hello world"
    assert result[0]["frame_count"] == 1


def test_identical_text_merges():
    """Identical text across frames merges into one span."""
    entries = [
        (1.0, "Introduction to Python", 0.90),
        (3.0, "Introduction to Python", 0.92),
        (5.0, "Introduction to Python", 0.88),
    ]
    result = deduplicate_ocr_results(entries)
    assert len(result) == 1
    assert result[0]["start_time"] == 1.0
    assert result[0]["end_time"] == 5.0
    assert result[0]["confidence"] == 0.92  # highest
    assert result[0]["frame_count"] == 3


def test_similar_multiline_merges():
    """Multi-line text with shared lines merges (high Jaccard)."""
    entries = [
        (1.0, "Title Bar\nSlide 1: Introduction\nFooter", 0.90),
        (3.0, "Title Bar\nSlide 1: Introduction\nFooter text", 0.85),
    ]
    # 2 out of 3/4 lines overlap -> Jaccard ~0.5-0.67
    result = deduplicate_ocr_results(entries, threshold=0.5)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.90


def test_chapter_numbers_split():
    """Short texts differing only in number must NOT merge.

    This was the key Codex finding: SequenceMatcher("Chapter 1", "Chapter 2")
    gives ~0.89 ratio (false merge). Line-level Jaccard gives 0.0 (correct).
    """
    entries = [
        (1.0, "Chapter 1: Introduction", 0.90),
        (10.0, "Chapter 2: Data Structures", 0.92),
    ]
    result = deduplicate_ocr_results(entries)
    assert len(result) == 2
    assert result[0]["text"] == "Chapter 1: Introduction"
    assert result[1]["text"] == "Chapter 2: Data Structures"


def test_mixed_merge_and_split():
    """Three frames: first two merge, third is different."""
    entries = [
        (1.0, "Welcome to the course", 0.90),
        (2.0, "Welcome to the course", 0.88),
        (10.0, "Now let us begin", 0.95),
    ]
    result = deduplicate_ocr_results(entries)
    assert len(result) == 2
    assert result[0]["frame_count"] == 2
    assert result[0]["end_time"] == 2.0
    assert result[1]["start_time"] == 10.0


def test_empty_text_entries_skipped():
    """Entries with empty or whitespace-only text are filtered out."""
    entries = [
        (1.0, "", 0.0),
        (2.0, "   ", 0.0),
        (3.0, "Actual text", 0.90),
    ]
    result = deduplicate_ocr_results(entries)
    assert len(result) == 1
    assert result[0]["text"] == "Actual text"


def test_max_gap_prevents_distant_merge():
    """Similar text separated by large time gap creates separate spans."""
    entries = [
        (1.0, "Recurring watermark", 0.90),
        (120.0, "Recurring watermark", 0.92),  # 119s gap
    ]
    result = deduplicate_ocr_results(entries, max_gap_seconds=30.0)
    assert len(result) == 2


def test_max_gap_allows_close_merge():
    """Similar text within time gap merges normally."""
    entries = [
        (1.0, "Recurring watermark", 0.90),
        (10.0, "Recurring watermark", 0.92),  # 9s gap
    ]
    result = deduplicate_ocr_results(entries, max_gap_seconds=30.0)
    assert len(result) == 1


def test_custom_threshold():
    """Custom threshold changes merge sensitivity."""
    entries = [
        (1.0, "Line A\nLine B\nLine C", 0.90),
        (2.0, "Line A\nLine D\nLine E", 0.85),
    ]
    # Jaccard = 1/5 = 0.2 (only "line a" shared)
    # Strict threshold -> split
    result_strict = deduplicate_ocr_results(entries, threshold=0.5)
    assert len(result_strict) == 2

    # Loose threshold -> merge
    result_loose = deduplicate_ocr_results(entries, threshold=0.1)
    assert len(result_loose) == 1


def test_normalize_line():
    """Normalization collapses whitespace and lowercases."""
    assert _normalize_line("  Hello   World  ") == "hello world"
    assert _normalize_line("UPPER CASE") == "upper case"


def test_to_line_set():
    """Splits text into normalized non-empty lines."""
    lines = _to_line_set("Hello\n\nWorld\n  Foo  ")
    assert lines == {"hello", "world", "foo"}


def test_jaccard_identical():
    """Identical sets have Jaccard 1.0."""
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    """Disjoint sets have Jaccard 0.0."""
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_empty():
    """Two empty sets have Jaccard 1.0, one empty has 0.0."""
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard(set(), {"a"}) == 0.0
```

**Step 3: Run tests**

```bash
cd /home/amlucas/dev/yt-llm-service
PYTHONPATH=src:tests/stubs:. uv run pytest tests/test_ocr_dedup.py -v
```

Expected: All 15 tests PASS.

**Step 4: Commit**

```bash
git add src/ocr_dedup.py tests/test_ocr_dedup.py
git commit -m "feat: add post-OCR text deduplication (line-level Jaccard merge)"
```

---

### Task 8: Add OCR endpoints to run_llm_api.py

> **Codex review finding:** `AudioDownloader.download_audio()` returns `audio_path` only, not `video_path`. The `/ocr-youtube` endpoint must download video separately using yt-dlp with a unique temp filename to avoid concurrency collisions.

**Files:**
- Modify: `src/run_llm_api.py`

**Step 1: Add imports and initialize OCR client**

Near the top of `run_llm_api.py`, alongside existing service imports:

```python
from frame_extractor import extract_scene_frames, cleanup_frames
from ocr_client import OCRClient
from ocr_dedup import deduplicate_ocr_results
```

In the initialization section (where `notes_service` is created):

```python
ocr_client = OCRClient(
    base_url=config.OCR_SERVICE_URL,
    timeout=config.OCR_SERVICE_TIMEOUT,
)
```

**Step 2: Add request/response models**

Add alongside existing Pydantic models:

```python
class YouTubeOCRRequest(BaseModel):
    youtube_url: str
    scene_threshold: float = 0.3
    max_frames: int = 100
    lang: str = "en"


class OCRSpan(BaseModel):
    start_time: float
    end_time: float
    text: str
    confidence: float
    frame_count: int


class OCRResponse(BaseModel):
    success: bool
    spans: list[OCRSpan]
    total_spans: int
    total_frames_processed: int
    processing_time_ms: float
    video_metadata: Optional[dict] = None
    error: Optional[str] = None
```

**Step 3: Add `/ocr-youtube` endpoint**

```python
@app.post("/ocr-youtube", response_model=OCRResponse)
async def ocr_youtube(request: YouTubeOCRRequest):
    """Extract text from YouTube video frames using OCR."""
    start = time.perf_counter()
    logger.info(f"OCR request for YouTube URL: {request.youtube_url}")

    video_path = None
    frames = []
    try:
        # Extract video_id and metadata via AudioDownloader (for VideoContext)
        video_id = audio_downloader._extract_video_id(request.youtube_url)
        video_context = audio_downloader.get_video_context(request.youtube_url)

        # Download video to a unique temp file (AudioDownloader only returns
        # audio_path, so we download video separately for frame extraction)
        import subprocess
        import uuid
        video_path = Path(config.TEMP_DIR) / f"ocr_{video_id}_{uuid.uuid4().hex[:8]}.mp4"
        dl_result = subprocess.run([
            "yt-dlp", "-f", "bestvideo[height<=1080][ext=mp4]/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(video_path), request.youtube_url,
        ], capture_output=True, text=True, timeout=120)
        if dl_result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {dl_result.stderr[:200]}")

        # Extract frames
        frames = extract_scene_frames(
            str(video_path),
            scene_threshold=request.scene_threshold,
            max_frames=request.max_frames,
        )

        if not frames:
            return OCRResponse(
                success=True, spans=[], total_spans=0, total_frames_processed=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                video_metadata=_video_context_to_dict(video_context) if video_context else None,
            )

        # Send to OCR service
        image_paths = [f.path for f in frames]
        ocr_result = await ocr_client.ocr_images(image_paths, lang=request.lang)

        if ocr_result is None:
            return OCRResponse(
                success=False, spans=[], total_spans=0, total_frames_processed=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                error="ocr-service unavailable",
            )

        # Combine frame timestamps with OCR results, then deduplicate
        raw_entries = [
            (frame.timestamp_s, ocr_frame.full_text, ocr_frame.avg_confidence)
            for frame, ocr_frame in zip(frames, ocr_result.results)
        ]
        spans = deduplicate_ocr_results(raw_entries)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResponse(
            success=True,
            spans=[OCRSpan(**s) for s in spans],
            total_spans=len(spans),
            total_frames_processed=len(frames),
            processing_time_ms=elapsed_ms,
            video_metadata=_video_context_to_dict(video_context) if video_context else None,
        )

    except Exception as e:
        logger.error(f"OCR YouTube error: {e}")
        return OCRResponse(
            success=False, spans=[], total_spans=0, total_frames_processed=0,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )
    finally:
        # Always clean up temp files
        if frames:
            cleanup_frames(frames)
        if video_path and Path(video_path).exists():
            Path(video_path).unlink(missing_ok=True)


def _video_context_to_dict(ctx) -> dict:
    """Convert VideoContext to a plain dict for the response."""
    return {
        "video_id": ctx.video_id,
        "title": ctx.title,
        "channel": ctx.channel,
        "tags": ctx.tags,
        "categories": ctx.categories,
    }
```

**Step 4: Add `/ocr-file` endpoint**

```python
@app.post("/ocr-file", response_model=OCRResponse)
async def ocr_file(
    file: UploadFile = File(...),
    scene_threshold: float = Form(0.3),
    max_frames: int = Form(100),
    lang: str = Form("en"),
):
    """Extract text from uploaded video file frames using OCR."""
    start = time.perf_counter()
    logger.info(f"OCR request for uploaded file: {file.filename}")

    upload_path = None
    frames = []
    try:
        # Save uploaded file with unique name
        import uuid
        safe_name = Path(file.filename).stem[:50]
        upload_path = Path(config.TEMP_DIR) / f"ocr_upload_{safe_name}_{uuid.uuid4().hex[:8]}.mp4"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract frames
        frames = extract_scene_frames(
            str(upload_path),
            scene_threshold=scene_threshold,
            max_frames=max_frames,
        )

        if not frames:
            return OCRResponse(
                success=True, spans=[], total_spans=0, total_frames_processed=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Send to OCR service
        image_paths = [f.path for f in frames]
        ocr_result = await ocr_client.ocr_images(image_paths, lang=lang)

        total_frames = len(frames)

        if ocr_result is None:
            return OCRResponse(
                success=False, spans=[], total_spans=0, total_frames_processed=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                error="ocr-service unavailable",
            )

        # Combine and deduplicate
        raw_entries = [
            (frame.timestamp_s, ocr_frame.full_text, ocr_frame.avg_confidence)
            for frame, ocr_frame in zip(frames, ocr_result.results)
        ]
        spans = deduplicate_ocr_results(raw_entries)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResponse(
            success=True,
            spans=[OCRSpan(**s) for s in spans],
            total_spans=len(spans),
            total_frames_processed=total_frames,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"OCR file error: {e}")
        return OCRResponse(
            success=False, spans=[], total_spans=0, total_frames_processed=0,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )
    finally:
        # Always clean up temp files
        if frames:
            cleanup_frames(frames)
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
```

**Step 5: Commit**

```bash
git add src/run_llm_api.py
git commit -m "feat: add /ocr-youtube and /ocr-file endpoints with dedup"
```

---

### Task 9: Integration test — build and run end-to-end

**Step 1: Build all services**

```bash
docker compose build ocr-service
docker compose up -d ocr-service
```

Wait for health check:
```bash
curl -s http://localhost:8003/health | python3 -m json.tool
```

**Step 2: Test with a YouTube video**

```bash
curl -X POST http://localhost:8002/ocr-youtube \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://youtu.be/MW3t6jP9AOs", "max_frames": 5}' \
  | python3 -m json.tool | head -40
```

Expected: JSON with `success: true`, `spans` array with `start_time`, `end_time`, `text`, `confidence`, `frame_count` per span. Repeated text across consecutive frames should be merged into single spans.

**Step 3: Test with file upload**

Use a test video file:
```bash
curl -X POST http://localhost:8002/ocr-file \
  -F "file=@/tmp/ocr-poc-video/source.mp4" \
  -F "max_frames=5" \
  | python3 -m json.tool | head -40
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration adjustments for OCR endpoints"
```

---

## File Summary

| File | Purpose |
|------|---------|
| `ocr-service/Dockerfile` | PyTorch CUDA + PaddleOCR container |
| `ocr-service/requirements.txt` | Sidecar Python deps |
| `ocr-service/app.py` | FastAPI sidecar (POST /ocr, GET /health) |
| `src/frame_extractor.py` | ffmpeg scene change detection |
| `src/ocr_client.py` | Async HTTP client for OCR sidecar |
| `src/ocr_dedup.py` | Post-OCR line-level Jaccard deduplication |
| `src/config.py` | OCR env var config (modified) |
| `src/run_llm_api.py` | /ocr-youtube, /ocr-file endpoints (modified) |
| `docker-compose.yml` | ocr-service added (modified) |
| `tests/test_frame_extractor.py` | Frame extraction unit tests |
| `tests/test_ocr_client.py` | OCR client unit tests |
| `tests/test_ocr_dedup.py` | Deduplication unit tests (15 tests) |
