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

### Task 7: Add OCR endpoints to run_llm_api.py

**Files:**
- Modify: `src/run_llm_api.py`

**Step 1: Add imports and initialize OCR client**

Near the top of `run_llm_api.py`, alongside existing service imports:

```python
from frame_extractor import extract_scene_frames, cleanup_frames
from ocr_client import OCRClient
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


class FrameOCR(BaseModel):
    timestamp_s: float
    text: str
    confidence: float
    frame_index: int


class OCRResponse(BaseModel):
    success: bool
    frames: list[FrameOCR]
    total_frames: int
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

    try:
        # Download video
        download_result = audio_downloader.download_audio(request.youtube_url)
        video_context = download_result.get("video_context")

        # Find the video file (yt-dlp downloads to temp dir)
        video_path = download_result.get("video_path")
        if not video_path:
            # Fallback: download video separately
            import subprocess
            video_path = str(Path(config.TEMP_DIR) / "ocr_video.mp4")
            subprocess.run([
                "yt-dlp", "-f", "bestvideo[height<=1080][ext=mp4]/best[height<=1080]",
                "--merge-output-format", "mp4",
                "-o", video_path, request.youtube_url,
            ], capture_output=True, timeout=120)

        # Extract frames
        frames = extract_scene_frames(
            video_path,
            scene_threshold=request.scene_threshold,
            max_frames=request.max_frames,
        )

        if not frames:
            return OCRResponse(
                success=True, frames=[], total_frames=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                video_metadata=_video_context_to_dict(video_context) if video_context else None,
            )

        # Send to OCR service
        image_paths = [f.path for f in frames]
        ocr_result = await ocr_client.ocr_images(image_paths, lang=request.lang)

        if ocr_result is None:
            cleanup_frames(frames)
            return OCRResponse(
                success=False, frames=[], total_frames=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                error="ocr-service unavailable",
            )

        # Combine frame timestamps with OCR results
        frame_ocrs = []
        for frame, ocr_frame in zip(frames, ocr_result.results):
            frame_ocrs.append(FrameOCR(
                timestamp_s=frame.timestamp_s,
                text=ocr_frame.full_text,
                confidence=ocr_frame.avg_confidence,
                frame_index=frame.index,
            ))

        cleanup_frames(frames)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResponse(
            success=True,
            frames=frame_ocrs,
            total_frames=len(frame_ocrs),
            processing_time_ms=elapsed_ms,
            video_metadata=_video_context_to_dict(video_context) if video_context else None,
        )

    except Exception as e:
        logger.error(f"OCR YouTube error: {e}")
        return OCRResponse(
            success=False, frames=[], total_frames=0,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


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

    try:
        # Save uploaded file
        upload_path = Path(config.TEMP_DIR) / f"ocr_upload_{file.filename}"
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
            upload_path.unlink(missing_ok=True)
            return OCRResponse(
                success=True, frames=[], total_frames=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Send to OCR service
        image_paths = [f.path for f in frames]
        ocr_result = await ocr_client.ocr_images(image_paths, lang=lang)

        cleanup_frames(frames)
        upload_path.unlink(missing_ok=True)

        if ocr_result is None:
            return OCRResponse(
                success=False, frames=[], total_frames=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                error="ocr-service unavailable",
            )

        frame_ocrs = [
            FrameOCR(
                timestamp_s=frame.timestamp_s,
                text=ocr_frame.full_text,
                confidence=ocr_frame.avg_confidence,
                frame_index=frame.index,
            )
            for frame, ocr_frame in zip(frames, ocr_result.results)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResponse(
            success=True,
            frames=frame_ocrs,
            total_frames=len(frame_ocrs),
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"OCR file error: {e}")
        return OCRResponse(
            success=False, frames=[], total_frames=0,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )
```

**Step 5: Commit**

```bash
git add src/run_llm_api.py
git commit -m "feat: add /ocr-youtube and /ocr-file endpoints"
```

---

### Task 8: Integration test — build and run end-to-end

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

Expected: JSON with `success: true`, `frames` array with `text`, `confidence`, `timestamp_s` per frame.

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
| `src/config.py` | OCR env var config (modified) |
| `src/run_llm_api.py` | /ocr-youtube, /ocr-file endpoints (modified) |
| `docker-compose.yml` | ocr-service added (modified) |
| `tests/test_frame_extractor.py` | Frame extraction unit tests |
| `tests/test_ocr_client.py` | OCR client unit tests |
