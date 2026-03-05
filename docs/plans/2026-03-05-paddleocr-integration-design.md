# PaddleOCR Integration Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan from this design.

## Goal

Add standalone OCR endpoints to yt-llm-service that extract text from video frames using PaddleOCR. Deployed as a thin sidecar container on GPU 1.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary goal | Standalone OCR endpoint | Independent of transcription pipeline |
| GPU | GPU 1 (RTX 3090 Ti, 24GB) | Enough VRAM for PaddleOCR (5GB) alongside llama-cpp (16.8GB with idle unload) |
| Input | YouTube URL + file upload | Mirrors existing transcription endpoint pair |
| Frame extraction | ffmpeg scene change detection | Best signal-to-noise ratio; skip talking head frames, capture slide changes |
| Container | Separate sidecar | Avoids PaddlePaddle/PyTorch CUDA runtime conflicts. Mirrors llama-cpp sidecar pattern |
| Architecture | Thin sidecar | OCR service is "images in, text out". Main service orchestrates download + frame extraction |

## Architecture

```
yt-llm-service :8002 (GPU 0)         ocr-service :8003 (GPU 1)
┌─────────────────────────┐          ┌──────────────────────┐
│                         │          │                      │
│ POST /ocr-youtube       │          │ POST /ocr            │
│   1. yt-dlp download    │          │   Input: N images    │
│   2. ffmpeg scene det   │  HTTP    │   Output: texts +    │
│   3. POST images ───────│─────────>│     confidences      │
│   4. Format response    │<─────────│                      │
│                         │          │ GET /health           │
│ POST /ocr-file          │          │   Output: status +   │
│   1. Save upload        │          │     GPU info         │
│   2. ffmpeg scene det   │          │                      │
│   3. POST images ───────│─────────>│ PaddleOCR v3         │
│   4. Format response    │<─────────│ (PP-OCRv5 server)    │
│                         │          │                      │
└─────────────────────────┘          └──────────────────────┘
```

## OCR Sidecar API

### POST /ocr

Accepts multipart images, returns OCR results per image.

**Request:** `multipart/form-data`
- `images[]`: List of image files (PNG/JPG)
- `lang`: string (default `"en"`)

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "texts": ["line 1", "line 2"],
      "confidences": [0.98, 0.95],
      "full_text": "line 1\nline 2",
      "avg_confidence": 0.965
    }
  ],
  "engine": "paddleocr",
  "processing_time_ms": 1234
}
```

### GET /health
```json
{
  "status": "ok",
  "engine": "paddleocr",
  "gpu": "NVIDIA GeForce RTX 3090 Ti",
  "vram_used_mb": 4200,
  "vram_total_mb": 24564
}
```

## Frame Extraction

New module `src/frame_extractor.py` in yt-llm-service.

Uses ffmpeg scene change detection:
```bash
ffmpeg -i video.mp4 \
  -vf "select='gt(scene,0.3)',showinfo" \
  -vsync vfr \
  frame_%04d.png
```

- Threshold: configurable (default 0.3)
- Safety cap: 100 frames max
- Output: list of `(frame_path, timestamp_s)` tuples
- Cleanup: delete temp frames after OCR completes

Parses ffmpeg `showinfo` filter output to extract PTS timestamps for each selected frame.

## Main Service Endpoints

### POST /ocr-youtube

```python
class YouTubeOCRRequest(BaseModel):
    youtube_url: str
    scene_threshold: float = 0.3
    max_frames: int = 100
    lang: str = "en"
```

### POST /ocr-file

Multipart form:
- `file`: UploadFile (required)
- `scene_threshold`: float = 0.3
- `max_frames`: int = 100
- `lang`: str = "en"

### Response (both endpoints)

```python
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
    video_metadata: Optional[dict]  # title, channel etc. (YouTube only)
    error: Optional[str]
```

## New Files

```
ocr-service/
  Dockerfile          # PyTorch 2.5 + PaddleOCR (from POC, proven)
  requirements.txt    # paddlepaddle-gpu, paddleocr, fastapi, uvicorn, pynvml
  app.py              # FastAPI sidecar (~100 lines)

src/
  frame_extractor.py  # ffmpeg scene change detection + frame extraction
  ocr_client.py       # Async HTTP client for ocr-service sidecar
```

## Docker Compose Addition

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
```

No dependency on llama-cpp or yt-llm-service. Independent lifecycle.

## Error Handling

- **ocr-service down**: Return error in OCRResponse (non-fatal)
- **No frames extracted**: Return empty frames list with `success: true`
- **Individual frame OCR failure**: Skip frame, log warning, continue with remaining
- **Timeout**: 60s per batch (configurable via `OCR_SERVICE_TIMEOUT` env var)

## Config

New env vars in yt-llm-service:
- `OCR_SERVICE_URL`: default `http://ocr-service:8003`
- `OCR_SERVICE_TIMEOUT`: default `60` (seconds)
- `OCR_SCENE_THRESHOLD`: default `0.3`
- `OCR_MAX_FRAMES`: default `100`
