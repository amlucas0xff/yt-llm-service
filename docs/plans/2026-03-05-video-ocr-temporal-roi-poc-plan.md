# Video OCR Temporal ROI POC -- Implementation Plan

**Goal:** Replace the current single-frame CLIP crop strategy with a video-aware OCR pipeline that uses box geometry, temporal stability, and text aggregation to produce materially better OCR from screencast YouTube videos.

**Why this POC:** The prior CLIP tile-cropping POC is operational but fails quality targets on real YouTube screencasts. Research and production tools converge on temporal methods (tracking + aggregation), plus constrained ROI.

**Scope:** POC only (`poc/`), no production integration yet.

**Target input for validation:** `https://www.youtube.com/watch?v=MW3t6jP9AOs`

---

## Evidence-Informed Design (what others are doing)

1. Video text spotting SOTA emphasizes **tracking + rescoring**, not frame-isolated OCR.
2. OCR engines already expose text polygons/confidences (`dt_polys`, `rec_polys`, `rec_scores`) and these should drive region filtering.
3. Practical subtitle/video OCR tools improve quality with **ROI constraints**, **frame dedupe (SSIM-like)**, and multi-zone support.
4. Border/canvas detection can be bootstrapped with FFmpeg crop tools before OCR.

---

## Architecture

1. Frame sampler: pull frames at configurable FPS and key moments.
2. Box-aware OCR: use polygon/box outputs, not only `full_text`.
3. Temporal ROI estimator: identify stable text-dense region across frames.
4. Track-and-aggregate: cluster boxes across time, aggregate recognition by confidence.
5. Output cleaner: suppress short/noisy lines and emit ordered text blocks.
6. Metrics reporter: compare baseline vs new workflow on line quality and noise rate.

---

## Task 1: Surface Box Geometry in OCR Path

**Files:**
- Update: `poc/ocr_helper.py`
- Add/Update tests: `poc/test_ocr_helper.py`

**Steps:**
1. Add helper returning normalized OCR objects with:
   - `text`, `score`, `poly`, `bbox`, `frame_index`, `timestamp`.
2. Support current sidecar response and direct Paddle structure when available.
3. Keep existing `ocr_image(...)` API for backward compatibility.
4. Add tests for:
   - parsing geometry fields,
   - fallback when only `full_text` exists.

**Validation:**
- `cd poc && uv run pytest test_ocr_helper.py -v`

---

## Task 2: Temporal ROI Estimator

**Files:**
- Add: `poc/temporal_roi.py`
- Add tests: `poc/test_temporal_roi.py`

**Steps:**
1. Build ROI candidates per frame from OCR box density and confidence.
2. Apply priors suitable for screencasts:
   - reject extreme margins/overlays,
   - favor persistent regions across frames,
   - optional center-right bias toggle (editor-heavy layouts).
3. Smooth ROI over time (EMA or rolling median).
4. Add optional manual seed ROI:
   - CLI arg: `--roi x1,y1,x2,y2`
   - behavior: intersect/union strategy controlled by flag.

**Validation:**
- `cd poc && uv run pytest test_temporal_roi.py -v`

---

## Task 3: Lightweight Text Tracking + Aggregation

**Files:**
- Add: `poc/text_tracker.py`
- Add tests: `poc/test_text_tracker.py`

**Steps:**
1. Match text boxes frame-to-frame using IoU + normalized text similarity.
2. Create track IDs for persistent text regions.
3. Aggregate each track:
   - weighted majority vote by OCR confidence,
   - keep highest-confidence canonical string,
   - retain start/end timestamps.
4. Drop transient tracks (min duration / min observations).

**Validation:**
- `cd poc && uv run pytest test_text_tracker.py -v`

---

## Task 4: New End-to-End POC Script

**Files:**
- Add: `poc/video_ocr_temporal.py`

**CLI:**
```bash
uv run python video_ocr_temporal.py <youtube_or_file> \
  --fps 1.5 \
  --roi-auto \
  --roi-stability-window 8 \
  --min-track-frames 3 \
  --rec-score-thresh 0.55 \
  --output-dir /tmp/video_ocr_temporal
```

**Steps:**
1. Reuse existing download/frame extraction helpers where possible.
2. Run baseline path (full-frame per sampled frame) and new temporal path.
3. Persist debug artifacts:
   - sampled frames,
   - ROI overlays,
   - per-frame OCR boxes,
   - track timeline JSON,
   - final merged text.

---

## Task 5: Quality Evaluation Harness

**Files:**
- Add: `poc/evaluate_temporal_ocr.py`

**Metrics (POC-level):**
1. Noise ratio: lines with low alnum density / gibberish heuristics.
2. Unique meaningful lines.
3. Repetition rate.
4. Avg confidence of retained lines.
5. Optional manual spot-check file for reviewer verdict.

**Acceptance thresholds (for this target video):**
1. >= 50% reduction in noise ratio vs baseline full-frame OCR.
2. >= 2x increase in meaningful unique lines vs current CLIP-only workflow.
3. Repetition rate reduced by >= 40%.
4. Reviewer judgment: “useful” over “garbage” on output text sample.

---

## Task 6: Experiment Matrix (Required)

Run on the target YouTube video with this matrix:
1. FPS: `1.0`, `1.5`, `2.0`
2. `rec_score_thresh`: `0.45`, `0.55`, `0.65`
3. ROI mode:
   - auto only
   - manual seed only
   - auto + manual constrained
4. Track min frames: `2`, `3`, `5`

Record each run in `/tmp/video_ocr_temporal/summary.csv`.

---

## Task 7: Decision Gate

At the end of experiments:
1. If acceptance thresholds are met:
   - mark POC successful and propose production-scope design.
2. If not met:
   - stop CLIP-based branch,
   - propose next POC on detector upgrade (DB/EAST/MMOCR detector front-end) while keeping temporal tracker.

---

## Implementation Notes

1. Keep diffs focused in `poc/`.
2. Preserve current `clip_ocr.py` for comparison (do not delete).
3. Add one Learning Loop entry to `CLAUDE.md` if this POC requires multi-attempt fixes.

---

## Verification Commands

```bash
cd /home/amlucas/dev/yt-llm-service/poc
uv run pytest -q
uv run python video_ocr_temporal.py "https://www.youtube.com/watch?v=MW3t6jP9AOs" --fps 1.5 --roi-auto --output-dir /tmp/video_ocr_temporal
uv run python evaluate_temporal_ocr.py --run-dir /tmp/video_ocr_temporal
```

---

## References

1. PaddleOCR output fields (`dt_polys`, `rec_polys`, `rec_scores`): https://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/OCR.html
2. GoMatching (video text spotting via tracking + rescoring): https://arxiv.org/abs/2401.07080
3. GoMatching++ (continued evidence for tracking-centric VTS): https://arxiv.org/abs/2505.22228
4. Temporal clustering for video scene text detection: https://arxiv.org/abs/2011.09781
5. OpenCV text spotting APIs (DB/EAST detector support): https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
6. FFmpeg crop/cropdetect workflow: https://patches.ffmpeg.org/ffmpeg-filters.html
7. Practical video OCR heuristics (ROI + frame similarity skipping): https://github.com/timminator/VideOCR
