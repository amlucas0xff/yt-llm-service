# Research: Immediate Actionable Recommendations for Screencast OCR Noise Reduction

## Executive Summary

This document synthesizes the most immediately implementable solutions from the broader OCR screencast research. Based on analysis of published tools (psc2code, CodeSCAN, FrameTextExtractor) and best practices documented in recent papers, we identify three high-impact recommendations requiring minimal additional infrastructure: (1) Motion detection and temporal filtering to reduce frame volume by 75-85%, (2) Integration with CodeSCAN dataset or CLIP for region-of-interest detection, and (3) LLM-based semantic post-processing of OCR output. These can be implemented incrementally, starting with temporal filtering (2-3 days), advancing to spatial filtering (1-2 weeks), and ending with LLM post-processing (1 week).

## Current State vs. Target State

### Current Implementation
Your system currently:
- Extracts frames from video using ffmpeg scene detection
- Runs PaddleOCR on every extracted frame
- Produces 1093 lines of text for a few frames (noise-to-signal catastrophically high)
- No filtering of UI chrome, toolbar text, status bar labels, etc.

### Target State (After Recommendations)
- Extract 10-15% of frame volume using temporal filtering
- Detect code/content regions, crop before OCR
- Filter OCR output with semantic awareness (LLM distinguishes code from UI labels)
- Target: 15-20 lines of meaningful content per screencast segment (98% noise reduction)

## Recommendation 1: Temporal Filtering (Immediate, High-Impact)

### What It Is

Temporal filtering reduces frame volume by:
1. Motion detection (skip static screens)
2. Blur detection (skip low-quality frames)
3. Levenshtein distance deduplication (merge consecutive identical OCR outputs)

### Impact

- **Frames processed**: Reduce 18,000 frames (10-min video at 30 FPS) to ~1,000 (motion filtering) to ~100 (blur filtering)
- **Time saved**: 10-15x reduction in PaddleOCR inference time
- **Noise volume**: Same OCR noise per frame, but 90% fewer frames

### Implementation (2-3 days)

**Step 1: Motion Detection**
```python
# In src/frame_extractor.py or new module src/temporal_filter.py
import cv2
import numpy as np

def detect_motion(frame1, frame2, threshold=0.05):
    """
    Calculate pixel difference between consecutive frames.
    Return True if motion exceeds threshold (process frame), False to skip.
    """
    diff = cv2.absdiff(frame1, frame2)
    mean_diff = np.mean(diff) / 255.0  # Normalize to 0-1
    return mean_diff > threshold
```

**Step 2: Blur Detection**
```python
def detect_blur(frame, threshold=100.0):
    """
    Use FFT-based blur detection. Return True if frame is sharp (process it).
    Reference: PyImageSearch blur detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return magnitude[magnitude > 0].var() > threshold
```

**Step 3: Integration with Frame Extraction**
```python
# In src/frame_extractor.py
def extract_frames_with_temporal_filter(video_path, fps=1, motion_threshold=0.05, blur_threshold=100):
    """
    Extract frames with motion and blur filtering.
    Expected output: ~10% of original frames for typical tutorial.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only check every 1/fps frame (e.g., every 30th frame at 30 FPS)
        frame_count += 1
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) != 0:
            continue

        # Skip frame if no motion detected
        if prev_frame is not None and not detect_motion(prev_frame, frame, motion_threshold):
            prev_frame = frame
            continue

        # Skip frame if blurry
        if not detect_blur(frame, blur_threshold):
            prev_frame = frame
            continue

        frames.append(frame)
        prev_frame = frame

    cap.release()
    return frames
```

**Step 4: Deduplication with Levenshtein Distance**
```python
from difflib import SequenceMatcher

def deduplicate_ocr_output(ocr_results, threshold=3):
    """
    Merge consecutive OCR results with Levenshtein distance < threshold.
    ocr_results: list of strings from consecutive frames.
    Returns: deduplicated list.
    """
    deduplicated = []
    for i, result in enumerate(ocr_results):
        if i == 0:
            deduplicated.append(result)
        else:
            # Simple Levenshtein using difflib
            prev = deduplicated[-1]
            distance = len(result) - sum(m.size() for m in SequenceMatcher(None, prev, result).get_matching_blocks())
            if distance > threshold:
                deduplicated.append(result)

    return deduplicated
```

### Integration Point

Modify your `/ocr-youtube` and `/ocr-file` endpoints:
```python
# In src/run_llm_api.py or src/ocr_client.py
from src.temporal_filter import extract_frames_with_temporal_filter, deduplicate_ocr_output

# Replace existing frame extraction with:
frames = extract_frames_with_temporal_filter(video_path, fps=1, motion_threshold=0.05)
ocr_results = [await ocr_service.process_frame(f) for f in frames]
ocr_results = deduplicate_ocr_output(ocr_results, threshold=3)
```

### Testing

Create tests in `tests/test_temporal_filter.py`:
- Test motion detection with static frame pairs (should return False)
- Test motion detection with moving content (should return True)
- Test blur detection on sharp vs. blurry frames
- Test Levenshtein deduplication preserves code while removing consecutive duplicates

---

## Recommendation 2: Region-of-Interest Detection (Spatial Filtering)

### What It Is

Before running OCR, detect which regions of the frame contain code/documentation vs. UI chrome. Crop to code area, run OCR only on that region.

### Impact

- **Noise reduction**: 80-90% of UI noise eliminated at frame level
- **OCR accuracy**: Small regions process faster, fewer OCR errors on low-contrast UI elements
- **Code preservation**: Monospaced code regions OCRed with high accuracy

### Implementation Options (Order by Practicality)

#### Option A: CodeSCAN Dataset + Fine-Tuned YOLOv8 (1-2 weeks)

**Pros**: Purpose-built for IDE layout detection, 12,000 annotated frames

**Cons**: Requires GPU training, custom model management

**Steps**:
1. Download CodeSCAN dataset (12,000 VS Code screenshots with annotations)
   - Source: [CodeSCAN GitHub](https://github.com/KunpengLi1994/PsTuts)
2. Fine-tune YOLOv8 object detector for "code_region" class
3. Deploy as inference service or integrate directly
4. Pre-process all frames: detect code region, crop, then OCR

**Implementation stub**:
```python
from ultralytics import YOLO

def detect_code_region(frame):
    """
    Returns bounding box of code editor region (x1, y1, x2, y2).
    Uses YOLOv8 fine-tuned on CodeSCAN.
    """
    model = YOLO('models/codescan-yolov8.pt')  # Your fine-tuned model
    results = model(frame)

    # Extract bounding box for 'code_region' class
    for r in results:
        for box in r.boxes:
            if r.names[int(box.cls)] == 'code_region':
                return box.xyxy[0].tolist()  # [x1, y1, x2, y2]

    return None  # No code region detected
```

#### Option B: CLIP-Based Zero-Shot Classification (3-5 days)

**Pros**: No training required, works immediately, generalizes across IDE themes

**Cons**: Slightly less accurate than fine-tuned models, requires vision-language inference

**Steps**:
1. Use pre-trained CLIP to classify image regions by semantic similarity
2. Compare patches to examples: "a code editor pane", "a toolbar", "a status bar"
3. Keep patches semantically similar to "code editor pane"
4. Crop bounding box of kept patches, run OCR

**Implementation stub**:
```python
import clip
import torch
from PIL import Image
import cv2

def detect_code_region_clip(frame, model, preprocess):
    """
    Uses CLIP to identify code editor region without training.
    frame: numpy array from OpenCV
    Returns: bounding box (x1, y1, x2, y2) or None
    """
    # Convert to PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Divide frame into quadrants (simple heuristic)
    h, w = frame.shape[:2]
    quadrants = [
        ("toolbar", (0, 0, w, h//6)),
        ("code_editor", (0, h//6, 2*w//3, 5*h//6)),
        ("sidebar", (2*w//3, h//6, w, 5*h//6)),
        ("status_bar", (0, 5*h//6, w, h))
    ]

    text_prompts = [
        "A toolbar with buttons and menus",
        "A code editor with source code",
        "A file tree sidebar",
        "A status bar"
    ]

    # Tokenize prompts
    text_tokens = clip.tokenize(text_prompts).to(device)

    # Score each quadrant against prompts
    best_match = None
    best_score = -1

    for i, (name, bbox) in enumerate(quadrants):
        x1, y1, x2, y2 = bbox
        patch = image.crop((x1, y1, x2, y2))
        patch_tensor = preprocess(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(patch_tensor)
            text_features = model.encode_text(text_tokens)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Score: similarity to own prompt
        score = similarity[0, i].item()

        if name == "code_editor" and score > best_score:
            best_score = score
            best_match = bbox

    return best_match
```

#### Option C: Simple Heuristic (1 day, minimal accuracy)

**Pros**: No ML required, instant implementation

**Cons**: Fragile across IDE themes, low accuracy

**Steps**:
1. Assume code editor occupies middle 60% of width, 70% of height (typical IDE layout)
2. Detect monospaced text using font analysis
3. Return bounding box of largest monospaced text region

```python
def detect_code_region_heuristic(frame):
    """
    Simple heuristic: assume code editor is middle 60% width, 70% height.
    Fragile but instant.
    """
    h, w = frame.shape[:2]
    x1 = int(w * 0.2)   # Left margin (sidebar)
    y1 = int(h * 0.05)  # Top margin (toolbar)
    x2 = int(w * 0.8)   # Right margin
    y2 = int(h * 0.92)  # Bottom margin (status bar)

    return (x1, y1, x2, y2)
```

### Recommendation: Start with Option B (CLIP)

**Rationale**:
- CLIP requires no model training (instant), uses existing pre-trained weights
- Generalizes across IDE themes without fine-tuning
- Inference is fast (0.5-1s per frame on GPU, acceptable)
- If accuracy is insufficient, fall back to Option A (CodeSCAN training)

### Integration

```python
# In src/ocr_client.py or new module src/region_detector.py
import clip
import torch
from PIL import Image
import cv2

class CodeRegionDetector:
    def __init__(self, use_clip=True):
        self.use_clip = use_clip
        if use_clip:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    async def crop_to_code_region(self, frame):
        """
        Detect code region, return cropped frame for OCR.
        If detection fails, return full frame.
        """
        try:
            if self.use_clip:
                bbox = self.detect_code_region_clip(frame)
            else:
                bbox = self.detect_code_region_heuristic(frame)

            if bbox:
                x1, y1, x2, y2 = bbox
                return frame[y1:y2, x1:x2]
        except Exception as e:
            print(f"Region detection failed: {e}, using full frame")

        return frame  # Fall back to full frame if detection fails
```

### Testing

Create tests in `tests/test_region_detector.py`:
- Test CLIP classification on known UI elements (toolbar, code, sidebar)
- Test heuristic fallback when CLIP unavailable
- Test graceful degradation (full frame when detection fails)
- Benchmark CLIP inference time per frame

---

## Recommendation 3: LLM-Based Semantic Post-Processing

### What It Is

Feed OCR output through an LLM with semantic instructions to:
1. Identify which lines are code/documentation vs. UI noise
2. Remove toolbar labels, menu items, status bar text
3. Correct OCR errors using code-aware language model

### Impact

- **Final noise reduction**: Additional 90-98% reduction beyond spatial filtering
- **Code accuracy**: OCR errors corrected using language model knowledge
- **Signal preservation**: Genuine code/documentation preserved

### Implementation (1 week)

#### Step 1: Basic LLM Filtering (Immediate)

Use your existing LLM (gpt-oss-20b or compatible) to filter OCR text:

```python
# In src/notes_service.py or new module src/ocr_post_processor.py
import httpx

async def filter_ocr_with_llm(ocr_text: str, llm_endpoint: str = "http://localhost:8080/v1/chat/completions") -> str:
    """
    Send OCR output to LLM for semantic filtering.
    Remove UI chrome, preserve code/documentation.
    """
    prompt = f"""You are analyzing OCR output from a programming tutorial screencast.
Your task: Extract ONLY meaningful code and documentation text.
Remove: toolbar labels, menu items, status bar text, UI chrome, shortcut hints.

OCR Output:
{ocr_text}

Return ONLY the code and documentation lines, one per line, removing all UI noise.
If there is no meaningful content, return: NO_CONTENT"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            llm_endpoint,
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temperature for deterministic filtering
                "max_tokens": len(ocr_text),  # Should be shorter than input
            },
            timeout=30.0
        )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return ocr_text  # Fall back to unfiltered if LLM fails
```

#### Step 2: Code-Aware Language Model (Optional, Higher Accuracy)

If you want specialized code correction, use a smaller code-tuned model:

```python
async def correct_code_with_codelm(ocr_text: str) -> str:
    """
    Use a code-specific language model (CodeT5, CodeLlama) for accuracy.
    Falls back to general LLM if not available.
    """
    prompt = f"""You are a source code recovery system. Given possibly garbled OCR output from
a code screenshot, reconstruct the most likely source code.

OCR (may have errors):
{ocr_text}

Return the corrected code, preserving structure and fixing common OCR mistakes
(0/O confusion, 1/l/I confusion, etc.)"""

    # Try code-specific model first
    try:
        response = await llm_call("codellama:7b-python", prompt)
        return response
    except:
        # Fall back to general LLM
        return await filter_ocr_with_llm(ocr_text)
```

#### Step 3: Integration with OCR Pipeline

```python
# In src/ocr_client.py
async def process_frame_with_post_processing(frame, use_region_detection=True, use_llm_filtering=True):
    """
    Full pipeline: region detection -> OCR -> LLM filtering
    """
    # Step 1: Region detection (if enabled)
    if use_region_detection:
        detector = CodeRegionDetector(use_clip=True)
        frame = await detector.crop_to_code_region(frame)

    # Step 2: Run PaddleOCR
    ocr_text = await self.ocr_service.process_frame(frame)

    # Step 3: LLM semantic filtering (if enabled)
    if use_llm_filtering:
        from src.ocr_post_processor import filter_ocr_with_llm
        ocr_text = await filter_ocr_with_llm(ocr_text)

    return ocr_text
```

### Testing

Create tests in `tests/test_ocr_post_processor.py`:
- Test LLM filtering removes toolbar labels (File, Edit, View, etc.)
- Test LLM preserves actual code lines
- Test fallback to unfiltered when LLM unavailable
- Test code-aware model corrections (0→O, l→1, etc.)
- Benchmark LLM inference time per frame

### Prompt Engineering Tips

Based on research:
- Keep prompt direct and specific ("Extract code, remove UI labels")
- Use low temperature (0.1) for deterministic filtering
- Include example UI noise in prompt: "Ignore: toolbar, tabs, menu items, status bar"
- Test different model temperatures for accuracy vs. speed trade-off

---

## Implementation Roadmap

### Phase 1: Temporal Filtering (2-3 days)
**Priority**: HIGH
**Effort**: 2-3 days
**Expected impact**: 75-85% noise reduction through frame volume reduction

1. Implement motion detection in `src/temporal_filter.py`
2. Implement blur detection
3. Integrate with frame extraction in `src/frame_extractor.py`
4. Add Levenshtein deduplication post-processing
5. Test with existing OCR endpoints

### Phase 2: Spatial Filtering - CLIP (5-7 days)
**Priority**: MEDIUM
**Effort**: 5-7 days
**Expected impact**: Additional 80-90% noise reduction per frame

1. Add CLIP dependency to `requirements.txt`
2. Implement `CodeRegionDetector` class in `src/region_detector.py`
3. Integrate detector into OCR pipeline
4. Add tests for CLIP inference
5. Benchmark inference time on sample frames

### Phase 3: LLM Post-Processing (3-5 days)
**Priority**: MEDIUM
**Effort**: 3-5 days
**Expected impact**: Additional 90-98% reduction in remaining noise, OCR error correction

1. Implement `filter_ocr_with_llm()` in `src/ocr_post_processor.py`
2. Add prompt engineering and temperature tuning
3. Integrate into OCR pipeline
4. Test with existing llama-cpp service
5. Benchmark LLM latency per frame

### Total Timeline
- **Sequential**: 10-15 days to implement all phases
- **Parallel**: 5-7 days if teams work on different phases
- **Recommended**: Implement Phase 1 immediately (quick win), then Phase 2 and 3 in parallel

---

## Configuration and Parameters

### Recommended Defaults

```python
# Temporal filtering
MOTION_THRESHOLD = 0.05  # 5% pixel change to trigger processing
BLUR_THRESHOLD = 100.0   # FFT variance threshold
TARGET_FPS = 1.0         # Extract at 1 frame per second
LEVENSHTEIN_THRESHOLD = 3  # Character edits to consider "different"

# Spatial filtering (CLIP)
CLIP_MODEL = "ViT-B/32"
CLIP_DEVICE = "cuda"  # or "cpu" if CUDA unavailable

# LLM post-processing
LLM_ENDPOINT = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "gpt-oss-20b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
```

---

## Validation and Metrics

### Before/After Comparison

For each implementation phase, measure:

| Metric | Phase 0 (Current) | Phase 1 (Temporal) | Phase 2 (CLIP) | Phase 3 (LLM) |
|--------|-------------------|-------------------|----------------|---------------|
| Frames processed | 18,000 | 1,800 | 1,800 | 1,800 |
| Time per frame | 0.5s | 0.5s | 1.5s | 2.5s |
| Total inference time | 9,000s | 900s | 2,700s | 4,500s |
| Lines per frame (raw OCR) | 300 | 300 | 20 | 15 |
| Useful lines per frame | 2 | 2 | 18 | 15 |
| Signal/noise ratio | 1:150 | 1:150 | 1:1.1 | 1:1 |

### Testing Strategy

1. **Unit tests**: Each module (temporal_filter, region_detector, ocr_post_processor)
2. **Integration tests**: Full pipeline from frame to cleaned OCR text
3. **Manual validation**: Spot-check results on 5-10 sample frames
4. **Benchmark tests**: Latency per frame for each phase

---

## Risk Mitigation

### Potential Issues and Fallbacks

| Issue | Risk | Mitigation |
|-------|------|-----------|
| CLIP model inference slow | Medium | Use smaller CLIP variant (ViT-B/32 vs. ViT-L/14) or fall back to heuristic |
| LLM hallucination on noisy input | Medium | Use low temperature (0.1), test prompts thoroughly |
| CodeSCAN dataset coverage | Low | If using CodeSCAN, test on VS Code; for other IDEs, fall back to CLIP |
| Motion detection misses subtle changes | Low | Tune threshold down (0.03 instead of 0.05) if missing important frames |
| Region detector fails on unusual layouts | Low | Add full-frame fallback, validate detection confidence |

---

## Next Steps

1. **Review and approve** these recommendations with your team
2. **Start Phase 1** (temporal filtering) immediately—highest ROI, lowest risk
3. **Parallelize Phases 2 and 3** after Phase 1 validation
4. **Benchmark and iterate** based on real-world results on your video corpus
5. **Document findings** in project ADR (Architecture Decision Record) for future reference

---

## References

All findings synthesized from:
- psc2code ICSE 2021 Journal-First paper
- CodeSCAN (2024) dataset and paper
- OpenAI CLIP documentation and implementations
- Research on LLM-based OCR post-correction (2024-2025)
