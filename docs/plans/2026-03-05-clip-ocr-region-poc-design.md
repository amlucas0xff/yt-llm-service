# POC: CLIP Region Cropping for Screencast OCR

## Problem

PaddleOCR on full screencast frames produces 1093 lines of output with ~1% signal.
UI chrome (toolbars, tabs, sidebars, status bars, keyboard overlays) dominates the output.
The meaningful content (code/documentation in the editor pane) is buried in noise.

## Goal

Prove that CLIP-based spatial filtering can isolate the code editor region from UI chrome,
producing OCR output where >80% of lines are meaningful content.

## Approach

Standalone script (`poc/clip_ocr.py`) that:

1. Loads an image (screenshot or extracted video frame)
2. Uses CLIP ViT-B/32 on GPU 0 (RTX 4060) to score a grid of image regions
   against text prompts ("source code in editor", "toolbar", "sidebar", etc.)
3. Merges adjacent high-scoring tiles into a bounding box
4. Crops the image to that bounding box
5. Sends the cropped image to ocr-service at localhost:8003/ocr
6. Prints full-frame OCR vs cropped OCR side by side

### CLIP Region Detection Strategy

- Divide the frame into an NxN grid (start with 4x4 = 16 tiles)
- For each tile, compute CLIP similarity against candidate labels:
  - Positive: "source code in a text editor", "programming code", "documentation text"
  - Negative: "toolbar with icons", "browser tabs", "file explorer sidebar",
    "status bar", "virtual keyboard", "video player controls"
- Tiles scoring above threshold on positive labels -> part of content region
- Merge adjacent qualifying tiles into a single bounding box
- Crop with small padding margin

### Why CLIP

- Zero-shot: no training data needed
- Generalizes across IDE themes, layouts, and content types
- ViT-B/32 is small (~340MB) and fast (<10ms on GPU)
- If the small model fails, the visual distinction is too subtle for any approach

## Dependencies

Local venv (not in Docker):
- openai-clip
- torch + torchvision (CUDA)
- Pillow
- httpx (for ocr-service calls)

## Test Input

Primary: `/home/amlucas/Captura de tela 2026-03-05 135912.png`
(VS Code screencast frame with code editor, sidebar, toolbar, browser chrome)

## Success Criteria

- Cropped OCR output: <30 lines total
- >80% of output lines are meaningful code/documentation content
- Visual inspection: crop boundary correctly isolates the editor pane
- Comparison output clearly shows improvement over full-frame OCR

## Out of Scope

- Video processing / temporal filtering
- LLM post-processing cleanup
- Service integration / Docker changes
- Production hardening

## Hardware

- CLIP inference: GPU 0 (RTX 4060, 8GB) -- ViT-B/32 uses ~500MB VRAM
- OCR inference: GPU 1 via ocr-service container (already running)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Delivery format | Standalone script | Fast iteration, no Docker rebuilds |
| CLIP model | ViT-B/32 | Smallest, sufficient for POC |
| CLIP device | GPU 0 | Sub-10ms inference for fast iteration |
| OCR engine | Existing ocr-service sidecar | Same engine as production, no new deps |
| Grid size | 4x4 initial | 16 tiles balances granularity vs speed |
