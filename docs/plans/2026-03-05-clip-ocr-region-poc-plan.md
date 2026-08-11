# CLIP Region Cropping OCR POC -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standalone script that uses CLIP to crop screencast frames to the code editor region before OCR, proving spatial filtering eliminates UI chrome noise.

**Architecture:** Load CLIP ViT-B/32 on GPU 0, divide image into NxN grid tiles, score each tile against "code editor" vs "UI chrome" text prompts, merge high-scoring tiles into a bounding box, crop, send to ocr-service for OCR, compare full-frame vs cropped output.

**Tech Stack:** Python 3.11+, openai-clip, torch (CUDA), Pillow, httpx, uv

**Test image:** `/home/amlucas/Captura de tela 2026-03-05 135912.png`

**OCR service:** `POST http://localhost:8003/ocr` (multipart file upload, returns JSON with `results[].full_text`)

---

### Task 1: Set Up POC Environment

**Files:**
- Create: `poc/pyproject.toml`

**Step 1: Create poc directory and pyproject.toml**

```toml
[project]
name = "clip-ocr-poc"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1",
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "httpx",
    "pytest",
    "pytest-asyncio",
]

[tool.uv]
extra-index-url = ["https://download.pytorch.org/whl/cu124"]
```

**Step 2: Create the venv and install deps**

Run: `cd poc && uv sync`
Expected: venv created, all deps installed including CUDA torch

Note: If torch installs CPU-only, use:
`uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124`

**Step 3: Verify CLIP loads on GPU**

Run:
```bash
cd poc && uv run python -c "
import clip, torch
device = 'cuda:0'
model, preprocess = clip.load('ViT-B/32', device=device)
print(f'CLIP loaded on {device}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M')
"
```
Expected: `CLIP loaded on cuda:0, params=151M`

**Step 4: Commit**

```bash
git add poc/pyproject.toml
git commit -m "chore: add poc/ venv for CLIP region cropping experiment"
```

---

### Task 2: CLIP Grid Scorer

**Files:**
- Create: `poc/clip_scorer.py`
- Test: `poc/test_clip_scorer.py`

**Step 1: Write the failing test**

```python
# poc/test_clip_scorer.py
"""Tests for CLIP grid scoring."""
from PIL import Image
from clip_scorer import score_grid, merge_tiles_to_bbox


def test_score_grid_returns_correct_shape():
    """6x6 grid on a 600x600 image -> 36 scores."""
    img = Image.new("RGB", (600, 600), color="black")
    scores = score_grid(img, grid_size=6)
    assert scores.shape == (6, 6)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_merge_tiles_produces_valid_bbox():
    """Merge should return (x1, y1, x2, y2) within image bounds."""
    import numpy as np
    # Simulate: center 2x2 tiles score high on a 6x6 grid, edges low
    scores = np.full((6, 6), 0.1)
    scores[2:4, 2:4] = 0.9  # center block
    bbox = merge_tiles_to_bbox(scores, image_size=(600, 600), threshold=0.5, padding=0)
    x1, y1, x2, y2 = bbox
    # 6x6 grid on 600x600 -> tile=100px, cols 2-3 (200-400), rows 2-3 (200-400)
    assert x1 == 200
    assert y1 == 200
    assert x2 == 400
    assert y2 == 400


def test_merge_tiles_with_padding_clamps():
    """Padding should not exceed image bounds."""
    import numpy as np
    scores = np.ones((6, 6))  # all tiles qualify
    bbox = merge_tiles_to_bbox(scores, image_size=(600, 600), threshold=0.5, padding=50)
    x1, y1, x2, y2 = bbox
    assert x1 == 0
    assert y1 == 0
    assert x2 == 600
    assert y2 == 600
```

**Step 2: Run test to verify it fails**

Run: `cd poc && uv run pytest test_clip_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clip_scorer'`

**Step 3: Write the implementation**

```python
# poc/clip_scorer.py
"""Score image regions using CLIP to identify code editor areas."""

import clip
import numpy as np
import torch
from PIL import Image

# Labels for zero-shot classification
POSITIVE_LABELS = [
    "source code in a text editor",
    "programming code on screen",
    "documentation text in an editor",
]
NEGATIVE_LABELS = [
    "toolbar with icons and buttons",
    "browser tab bar",
    "file explorer sidebar",
    "terminal status bar",
    "virtual keyboard",
    "video player controls",
    "desktop taskbar",
]

_model = None
_preprocess = None
_text_features = None


def _get_model(device: str = "cuda:0"):
    """Lazy-load CLIP model and pre-encode text features."""
    global _model, _preprocess, _text_features
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device=device)
        all_labels = POSITIVE_LABELS + NEGATIVE_LABELS
        tokens = clip.tokenize(all_labels).to(device)
        with torch.no_grad():
            _text_features = _model.encode_text(tokens)
            _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
    return _model, _preprocess, _text_features


def score_grid(
    image: Image.Image,
    grid_size: int = 6,
    device: str = "cuda:0",
) -> np.ndarray:
    """Divide image into grid_size x grid_size tiles and score each.

    Returns a (grid_size, grid_size) array of scores in [0, 1].
    Score = average similarity to positive labels minus average similarity
    to negative labels, rescaled to [0, 1].
    """
    model, preprocess, text_features = _get_model(device)
    w, h = image.size
    tile_w, tile_h = w // grid_size, h // grid_size

    n_pos = len(POSITIVE_LABELS)
    scores = np.zeros((grid_size, grid_size), dtype=np.float32)

    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * tile_w
            y1 = row * tile_h
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            tile = image.crop((x1, y1, x2, y2))
            tiles.append(preprocess(tile))

    # Batch encode all tiles at once
    tile_batch = torch.stack(tiles).to(device)
    with torch.no_grad():
        image_features = model.encode_image(tile_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).cpu().numpy()

    for idx in range(grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        pos_score = similarity[idx, :n_pos].mean()
        neg_score = similarity[idx, n_pos:].mean()
        # Rescale difference from [-1, 1] to [0, 1]
        scores[row, col] = np.clip((pos_score - neg_score + 1) / 2, 0, 1)

    return scores


def merge_tiles_to_bbox(
    scores: np.ndarray,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    padding: int = 10,
) -> tuple[int, int, int, int]:
    """Merge qualifying tiles into a single bounding box.

    Args:
        scores: (rows, cols) array of tile scores
        image_size: (width, height) of the original image
        threshold: minimum score to include a tile
        padding: pixels to add around the merged region

    Returns:
        (x1, y1, x2, y2) bounding box in pixel coordinates.
        Falls back to full image if no tiles qualify.
    """
    rows, cols = scores.shape
    w, h = image_size
    tile_w, tile_h = w // cols, h // rows

    mask = scores >= threshold
    if not mask.any():
        return (0, 0, w, h)

    qualifying = np.argwhere(mask)  # (N, 2) array of (row, col)
    min_row, min_col = qualifying.min(axis=0)
    max_row, max_col = qualifying.max(axis=0)

    x1 = max(0, min_col * tile_w - padding)
    y1 = max(0, min_row * tile_h - padding)
    x2 = min(w, (max_col + 1) * tile_w + padding)
    y2 = min(h, (max_row + 1) * tile_h + padding)

    return (x1, y1, x2, y2)
```

**Step 4: Run tests to verify they pass**

Run: `cd poc && uv run pytest test_clip_scorer.py -v`
Expected: 3 passed

Note: `test_score_grid_returns_correct_shape` requires GPU. If running on a machine without GPU, skip it with `pytest -k "not score_grid_returns"`.

**Step 5: Commit**

```bash
git add poc/clip_scorer.py poc/test_clip_scorer.py
git commit -m "feat(poc): add CLIP grid scorer for code region detection"
```

---

### Task 3: OCR Client Helper

**Files:**
- Create: `poc/ocr_helper.py`
- Test: `poc/test_ocr_helper.py`

**Step 1: Write the failing test**

```python
# poc/test_ocr_helper.py
"""Tests for OCR helper (mocked HTTP)."""
import json
import httpx
import pytest
from ocr_helper import ocr_image


@pytest.fixture
def mock_ocr_response():
    return {
        "results": [{
            "index": 0,
            "texts": ["def hello():", "    print('world')"],
            "confidences": [0.95, 0.92],
            "full_text": "def hello():\n    print('world')",
            "avg_confidence": 0.935,
        }],
        "engine": "paddleocr",
        "processing_time_ms": 123.4,
    }


@pytest.mark.asyncio
async def test_ocr_image_returns_text(mock_ocr_response, tmp_path):
    """OCR helper should return full_text from service response."""
    # Create a dummy PNG
    from PIL import Image
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 100)).save(img_path)

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=mock_ocr_response)
    )
    result = await ocr_image(str(img_path), transport=transport)
    assert result == "def hello():\n    print('world')"


@pytest.mark.asyncio
async def test_ocr_image_returns_none_on_error(tmp_path):
    """OCR helper should return None if service errors."""
    from PIL import Image
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 100)).save(img_path)

    transport = httpx.MockTransport(
        lambda req: httpx.Response(500, text="Internal Server Error")
    )
    result = await ocr_image(str(img_path), transport=transport)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd poc && uv run pytest test_ocr_helper.py -v`
Expected: FAIL (need pytest-asyncio too: `uv add pytest pytest-asyncio`)

**Step 3: Write the implementation**

```python
# poc/ocr_helper.py
"""Thin async wrapper around the ocr-service HTTP API."""

import httpx

OCR_SERVICE_URL = "http://localhost:8003/ocr"


async def ocr_image(
    image_path: str,
    base_url: str = OCR_SERVICE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str | None:
    """Send an image to ocr-service and return the full_text.

    Returns None on any error.
    """
    kwargs = {"timeout": httpx.Timeout(60.0)}
    if transport:
        kwargs["transport"] = transport

    try:
        async with httpx.AsyncClient(**kwargs) as client:
            with open(image_path, "rb") as f:
                response = await client.post(
                    base_url,
                    files=[("images", (image_path.split("/")[-1], f, "image/png"))],
                    data={"lang": "en"},
                )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]["full_text"]
            return ""
    except Exception:
        return None
```

**Step 4: Run tests**

Run: `cd poc && uv run pytest test_ocr_helper.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add poc/ocr_helper.py poc/test_ocr_helper.py
git commit -m "feat(poc): add async OCR helper for ocr-service sidecar"
```

---

### Task 4: Main POC Script

**Files:**
- Create: `poc/clip_ocr.py`

This is the glue script. No unit test -- validated by running it and inspecting output.

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""POC: CLIP-based region cropping for screencast OCR.

Usage:
    uv run python clip_ocr.py <image_path> [--grid N] [--threshold T] [--save-crops]

Compares full-frame OCR vs CLIP-cropped OCR side by side.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from PIL import Image

from clip_scorer import merge_tiles_to_bbox, score_grid
from ocr_helper import ocr_image


def print_section(title: str, text: str | None, max_lines: int = 50):
    """Print a labeled section of OCR output."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if text is None:
        print("  [OCR service unavailable]")
        return 0
    lines = [l for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines[:max_lines]):
        print(f"  {i+1:3d} | {line}")
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} more lines)")
    return len(lines)


async def run(args):
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path.name} ({img.size[0]}x{img.size[1]})")

    # Step 1: Full-frame OCR (baseline)
    print("\nRunning full-frame OCR...")
    t0 = time.perf_counter()
    full_path = str(image_path)
    full_text = await ocr_image(full_path)
    t_full = time.perf_counter() - t0

    # Step 2: CLIP region detection
    print(f"\nScoring {args.grid}x{args.grid} grid with CLIP...")
    t0 = time.perf_counter()
    scores = score_grid(img, grid_size=args.grid, device=args.device)
    t_clip = time.perf_counter() - t0

    # Print score heatmap
    print(f"\nCLIP scores (threshold={args.threshold}):")
    for row in range(scores.shape[0]):
        cells = []
        for col in range(scores.shape[1]):
            s = scores[row, col]
            marker = "*" if s >= args.threshold else " "
            cells.append(f"{s:.2f}{marker}")
        print(f"  [{' | '.join(cells)}]")

    # Step 3: Merge and crop
    bbox = merge_tiles_to_bbox(
        scores, image_size=img.size,
        threshold=args.threshold, padding=args.padding,
    )
    x1, y1, x2, y2 = bbox
    print(f"\nCrop region: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"Crop size: {x2-x1}x{y2-y1} (original: {img.size[0]}x{img.size[1]})")
    crop_ratio = ((x2 - x1) * (y2 - y1)) / (img.size[0] * img.size[1])
    print(f"Area ratio: {crop_ratio:.1%} of original")

    cropped = img.crop(bbox)

    if args.save_crops:
        crop_path = image_path.with_stem(f"{image_path.stem}_cropped")
        cropped.save(crop_path)
        print(f"Saved crop: {crop_path}")

    # Step 4: Cropped OCR
    # Save crop to temp file for OCR service
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cropped.save(tmp.name)
        crop_tmp = tmp.name

    print("\nRunning cropped OCR...")
    t0 = time.perf_counter()
    crop_text = await ocr_image(crop_tmp)
    t_crop_ocr = time.perf_counter() - t0
    Path(crop_tmp).unlink(missing_ok=True)

    # Step 5: Compare
    n_full = print_section("FULL-FRAME OCR", full_text)
    n_crop = print_section("CROPPED OCR (CLIP region)", crop_text)

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Full-frame lines:  {n_full}")
    print(f"  Cropped lines:     {n_crop}")
    if n_full > 0:
        print(f"  Reduction:         {(1 - n_crop/n_full)*100:.0f}%")
    print(f"  CLIP scoring:      {t_clip*1000:.0f}ms")
    print(f"  Full-frame OCR:    {t_full*1000:.0f}ms")
    print(f"  Cropped OCR:       {t_crop_ocr*1000:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="CLIP region cropping OCR POC")
    parser.add_argument("image", help="Path to screenshot or video frame")
    parser.add_argument("--grid", type=int, default=6, help="Grid size NxN (default: 6)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold (default: 0.5)")
    parser.add_argument("--padding", type=int, default=10, help="Crop padding in pixels (default: 10)")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped image to disk")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add poc/clip_ocr.py
git commit -m "feat(poc): add main CLIP OCR comparison script"
```

---

### Task 5: Run the POC and Validate

**Prereqs:** ocr-service container must be running on localhost:8003.

**Step 1: Verify ocr-service is up**

Run: `curl -s http://localhost:8003/health | python3 -m json.tool`
Expected: `{"status": "ok", "engine": "paddleocr", ...}`

If not running: `cd /home/amlucas/dev/yt-llm-service && docker compose up ocr-service -d`

**Step 2: Run the POC on the test screenshot**

Run:
```bash
cd poc && uv run python clip_ocr.py \
  "/home/amlucas/Captura de tela 2026-03-05 135912.png" \
  --grid 6 --threshold 0.5 --save-crops
```

Expected output:
- CLIP score heatmap showing high scores in center (editor) and low on edges (chrome)
- Crop region covering the editor pane, not the toolbar/sidebar
- Full-frame OCR: ~hundreds of lines
- Cropped OCR: <30 lines, mostly code/documentation
- Reduction: >70%
- Saved cropped image for visual inspection

**Step 3: Inspect the cropped image**

Open the saved `*_cropped.png` file. Verify:
- The crop covers the code editor pane
- Toolbar, sidebar, and browser chrome are excluded
- Code text is fully visible and not cut off

**Step 4: Tune if needed**

If crop is too tight: lower threshold (e.g., `--threshold 0.4`)
If crop includes too much chrome: raise threshold (e.g., `--threshold 0.6`)
If tiles are too coarse: increase grid (e.g., `--grid 6` or `--grid 8`)

**Step 5: Record results and commit**

```bash
git add -A poc/
git commit -m "feat(poc): validated CLIP region cropping -- [X]% noise reduction"
```

Replace `[X]` with actual measured reduction.

---

### Success Criteria Checklist

- [ ] CLIP model loads on GPU 0 and scores tiles in <50ms
- [ ] Score heatmap visually correlates with code region
- [ ] Cropped image isolates the editor pane
- [ ] Cropped OCR output: <30 lines
- [ ] >80% of cropped OCR lines are meaningful content
- [ ] Side-by-side comparison shows clear improvement
