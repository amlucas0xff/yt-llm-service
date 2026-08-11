# Research: OCR Text Extraction from Programming Screencasts - Noise Problem and Solutions

## Executive Summary

Programming screencasts present a unique OCR challenge: when extracting text from IDE frames, OCR engines capture entire UI layers (toolbars, tabs, status bars, sidebars, keyboard overlays) with equal priority as meaningful code content, resulting in extremely low signal-to-noise ratios. This research synthesizes published approaches to this problem, covering four main solution categories: (1) region-of-interest detection and content area isolation, (2) temporal filtering using frame differencing and consistency checks, (3) LLM-based post-processing for noise filtering, and (4) specialized tools and datasets specifically designed for screencast analysis. The most promising approaches combine multiple techniques: motion detection to skip static frames, CNN-based code region detection, and code-aware language models for post-correction.

## Scope and Context

- **Research focus**: Techniques for extracting meaningful code/documentation text from programming tutorial video frames while filtering UI chrome, toolbars, status bars, and other non-content elements
- **Problem scope**: General OCR is 1093 lines of noise for a few frames; we need practical reduction to 10-20 lines of genuine content
- **Reference materials reviewed**: Academic papers, GitHub projects, technical blog posts, and commercial tools
- **Key search areas**: Region-of-interest detection, temporal filtering, code-specific OCR, LLM post-processing, and dataset resources

## Part 1: The General Problem - Why Screenshot OCR Captures Everything

### Problem Definition

[Screenshot OCR UI Noise Problem](https://www.pdnob.com/screen-translator/ocr-from-screenshot.html) documents that general OCR on screenshots automatically detects non-selectable UI elements (error messages, toolbar text, tab names, menu labels, keyboard keys, status bar indicators) with the same confidence as content. Modern screenshot OCR tools attempt automatic "UI detection" but this typically means de-duplication of common system fonts and UI patterns—not content-aware filtering.

### Industry Recognition

The challenge is explicitly documented in multiple professional contexts:

- **IronOCR's ReadScreenshot method**: Acknowledges that "automatic noise reduction & UI detection" is necessary when extracting from screenshots, but their solution applies generic contrast enhancement and system font detection, not content-area isolation.

- **NormCap** ([GitHub - dynobo/normcap](https://github.com/dynobo/normcap)): An OCR-powered screen-capture tool designed to "capture information instead of images," recognizing that raw OCR is insufficient for practical screenshot analysis.

- **Screenshot-to-Text Chrome Extension** ([Chrome Web Store - Screenshot to Text OCR](https://chromewebstore.google.com/detail/screenshot-to-textocr-ima/mpfhimfooelblffmiddjmpnpndeldikg?hl=en)): Popular tool that extracts text from any visible UI, with no built-in code/content awareness.

### Screencast-Specific Challenge

Research on code extraction from videos explicitly identifies this as a blocking problem:

[psc2code: Denoising Code Extraction from Programming Screencasts](https://xin-xia.github.io/publication/tosem201.pdf) and the [ICSE 2021 Journal-First Paper](https://2021.icse-conferences.org/details/icse-2021-Journal-First-Papers/54/psc2code-Denoising-Code-Extraction-from-Programming-Screencasts) document that "OCR is not precise enough to produce quality results from videos" due to the overlay problem: code is written on-the-fly with only portions visible, and OCR captures UI chrome indiscriminately.

## Part 2: Region-of-Interest (ROI) Detection and Content Area Isolation

### Academic Approach: Screenshot Understanding Taxonomy

[Turning Screenshots into Data: A Four-Level Taxonomy for Screenshot Understanding](https://medium.com/data-science-collective/turning-screensots-int-data-html-126bdcaa4821) provides a structured framework for automated screenshot analysis:

1. **Level 1**: Detect presence of UI components (buttons, text fields, images)
2. **Level 2**: Classify UI element types (navigation bar, code area, sidebar)
3. **Level 3**: Extract text content with bounding boxes from each region
4. **Level 4**: Understand semantic relationships between components

For IDE screenshots, the content area (code editor pane) is typically Level-2 identifiable by its:
- Monospaced font (Courier, Consolas, Monaco, etc.)
- Syntax highlighting patterns (colors different from UI chrome)
- Grid-aligned text layout (line numbers + code indentation)
- Absence of interactive elements (buttons, drop-downs)

### GUI Component Detection with Deep Learning

[How Do You Use Deep Learning to Identify UI Components?](https://www.alibabacloud.com/blog/how-do-you-use-deep-learning-to-identify-ui-components_597859) describes using object detection models (YOLO, SSD, Faster R-CNN) to identify specific UI regions. The Rico and ReDraw datasets provide annotated screenshots with 15+ UI element categories (RadioButton, ProgressBar, Switch, Button, Checkbox, EditText, etc.), enabling trained detectors to identify content-bearing regions vs. chromatic UI.

### Implementation Approach: psc2code Algorithm

The psc2code approach ([Research Paper](https://xin-xia.github.io/publication/tosem201.pdf)) provides a concrete method:

1. **CNN-based frame classification**: Removes non-code frames (title slides, credits, etc.) using a convolutional neural network trained on annotated screencast frames
2. **Edge detection + clustering**: Applies edge detection (Sobel or Canny) to identify sharply-defined regions (code text has high contrast edges)
3. **Connected-component analysis**: Groups related pixels into candidate code regions
4. **Bounding box extraction**: Crops detected code regions before OCR

This approach directly addresses the noise problem by working in three stages:
- Spatial filtering (detect code region geometry before character recognition)
- Temporal filtering (skip non-coding frames entirely)
- OCR-only on cropped regions (reduces noise footprint)

### CodeSCAN Dataset

[CodeSCAN: ScreenCast ANalysis for Video Programming Tutorials](https://a-nau.github.io/codescan/) and its [arXiv paper](https://arxiv.org/abs/2409.18556) provide a modern, large-scale resource: 12,000 annotated screenshots from Visual Studio Code with:

- 24 programming languages
- 25 different fonts and font sizes
- 90+ distinct IDE themes
- Pixel-level annotations for code regions vs. UI chrome

This dataset can be used to fine-tune object detectors specifically for IDE layout recognition.

## Part 3: Temporal Filtering and Frame Differencing

### Motion Detection to Skip Static Frames

[Video OCR - OCR Video to Text Online Free | Extract Text from Videos](https://screenapp.io/features/video-ocr) and [Comparing the best methods for OCR on videos](https://www.sieve.ai/blog/video-ocr-guide) document that **motion detection** dramatically reduces OCR operations:

- **Approach**: Calculate frame difference (pixel-level L2 distance or optical flow) between consecutive frames
- **Threshold**: If difference is below threshold (e.g., <5% pixel change), skip OCR
- **Benefit**: Tutorials often have static screens while the instructor talks; only OCR when content visibly changes
- **Implementation**: OpenCV's `cv2.absdiff()` or ffmpeg's `select` filter with motion detection

### Consistency Filtering with Levenshtein Distance

[How to Use OCR on Videos](https://blog.roboflow.com/ocr-on-videos/) describes **Levenshtein distance thresholding**:

- When OCR outputs from consecutive frames have a Levenshtein distance below a threshold (e.g., <3 character edits), treat them as identical
- Remove duplicate consecutive OCR outputs automatically
- Preserve repeated text that appears across multiple frames (important code patterns)
- Does NOT remove global duplicates, only consecutive ones (preserves intentional repetition in tutorials)

### Blur Detection to Filter Low-Quality Frames

[OCR'ing Video Streams - PyImageSearch](https://pyimagesearch.com/2022/03/07/ocring-video-streams/) documents **FFT-based blur detection**:

- Compute Fast Fourier Transform (FFT) of frame
- Low-frequency energy indicates blur (motion blur, out of focus)
- Frames below blur threshold can be skipped, improving OCR accuracy
- Example threshold: discard frames with blur variance below 100

### Frame Extraction Optimization

[How to Extract Text and OCR from Video Frames](https://oneuptime.com/blog/post/2026-02-17-how-to-extract-text-and-ocr-from-video-frames-with-video-intelligence-api-text-detection/view) provides practical guidance:

- Extract frames at 1 FPS (sufficient for tutorial content where screens are visible for several seconds)
- Apply motion detection first (skip ~80% of frames in typical tutorials)
- Apply blur detection on remaining frames (further reduce to high-quality frames)
- Run OCR only on ~10-15% of original frames
- Use Levenshtein distance to merge identical consecutive outputs

**Concrete example**: A 10-minute screencast at 30 FPS = 18,000 frames. With motion + blur filtering:
- 1 FPS extraction = 600 frames
- 80% skipped by motion detection = 120 frames
- 50% skipped by blur detection = 60 frames
- Process only 60 frames, then merge consecutive identical results with Levenshtein distance

## Part 4: Code-Specific OCR and Fine-Tuning

### Research Findings on Code Accuracy

[A Study on the Accuracy of OCR Engines for Source Code](https://par.nsf.gov/servlets/purl/10220037) and [We Fine-Tuned our OCR to Read Code: Here's What It Took (and What Broke)](https://dev.to/nikl/we-fine-tuned-our-ocr-to-read-code-heres-what-it-took-and-what-broke-4jb8) document that:

- Standard OCR (Tesseract, PaddleOCR) achieves 92-98% accuracy on general text
- Accuracy drops to 60-75% on source code due to:
  - Monospaced fonts with minimal visual distinction between similar characters (0/O, 1/l/I, 2/Z)
  - Syntax highlighting colors that OCR ignores
  - Special characters (brackets, braces, operators) that are context-dependent
- Pieces.app has fine-tuned OCR specifically for code with custom character-level dictionaries

### Comparative Accuracy: PaddleOCR vs. Tesseract

[PaddleOCR vs Tesseract: Which is the best open source OCR?](https://www.koncile.ai/en/ressources/paddleocr-analyse-avantages-alternatives-open-source) and [OCR comparison: Tesseract versus EasyOCR vs PaddleOCR vs MMOCR](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66):

- **PaddleOCR**: 96.58% accuracy on invoices, handles complex layouts better, GPU acceleration available
- **Tesseract**: 98%+ accuracy on clean documents, 6x faster on CPU, but struggles with rotated/irregular text
- **EasyOCR**: Moderate accuracy (90-95%), excellent for multi-language support, slower than PaddleOCR
- **Recommendation for code**: PaddleOCR with GPU acceleration due to better layout handling and code-aware post-processing potential

### Post-Processing for Code Correction

[How to Extract Code from A Screenshot with Accuracy?](https://rikkeisoft.com/blog/extract-code-from-screenshot/) and [Image to Code OCR Just Released!](https://mathpix.com/blog/image-to-code-ocr) document that:

- Pieces.app extracts code from screenshots with specialized preprocessing
- Mathpix's image-to-code tool includes layout inference and structure preservation
- CodeCapture Chrome extension uses Tesseract with region selection, then manual review

**Key finding**: Code extraction requires domain-aware post-processing, not just raw OCR.

## Part 5: LLM-Based Post-Processing for Noise Filtering

### Current Research on OCR Error Correction

[Reference-Based Post-OCR Processing with LLM for Precise Diacritic Text](https://arxiv.org/html/2410.13305v1) and [OCR Error Post-Correction with LLMs in Historical Documents](https://arxiv.org/html/2502.01205v1) demonstrate:

- Fine-tuned LLMs (like ByT5) can correct character-level OCR errors with 56% Character Error Rate reduction
- LLMs work by learning semantic coherence: if OCR produces "cIass" in a Python file, LLM corrects to "class"
- However: LLMs hallucinate on noisy/garbled input; error correction requires reference text or domain constraints

### Hybrid OCR-LLM Architecture

[Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction](https://arxiv.org/html/2510.10138v1) proposes:

1. **Multi-engine OCR**: Run both Tesseract and PaddleOCR, use ensemble confidence
2. **Heuristic filtering**: Remove watermarks, stamps, short text fragments that are likely UI chrome
3. **LLM semantic filtering**: Feed concatenated OCR output to LLM with prompt: "Extract only code/documentation text, remove UI labels and toolbar text"
4. **Code-aware LLM**: Use a code-tuned model (CodeT5, Codex) familiar with programming syntax

### Practical Implementation Patterns

[Using LLMs for OCR and PDF Parsing](https://www.cradl.ai/posts/llm-ocr) and [Large Language Models LLMs for OCR Post-Correction](https://www.marktechpost.com/2024/08/13/large-language-models-llms-for-ocr-post-correction/) document:

- **Two-stage approach**: Raw OCR → LLM filtering → cleaned text
- **Prompt engineering**: "From this IDE screenshot OCR output, extract only the code visible in the editor pane. Ignore: toolbar, tabs, status bar, menu items, file tree labels"
- **Confidence scoring**: LLM can assign confidence to each extracted line (high = likely code, low = likely UI noise)
- **Fine-tuning targets**: Code-specific models like CodeT5-OCRfix leverage large code corpora to recover garbled code

### CodeT5-OCRfix Approach

From the [extract code from screencast research](https://xin-xia.github.io/publication/tosem201.pdf) context, a recent approach uses pre-trained code models:

- CodeT5 trained on 10K+ GitHub Java projects can predict most-likely code given partial OCR
- Example: OCR produces "def my_funktion(x, y):", CodeT5 suggests "function" not "funktion"
- Works for common patterns but requires training/fine-tuning on target languages

## Part 6: Existing Tools and Projects

### Open-Source Projects

**psc2code** ([ICSE 2021 Journal-First](https://2021.icse-conferences.org/details/icse-2021-Journal-First-Papers/54/psc2code-Denoising-Code-Extraction-from-Programming-Screencasts), [PDF](https://xin-xia.github.io/publication/tosem201.pdf))
- Addresses screencast code extraction directly
- Three-stage pipeline: frame classification (CNN), code region detection (edge detection + clustering), OCR on cropped regions
- Achieves significant noise reduction compared to full-frame OCR

**CodeSCAN** ([Website](https://a-nau.github.io/codescan/), [GitHub - KunpengLi1994/PsTuts](https://github.com/KunpengLi1994/PsTuts), [arXiv](https://arxiv.org/abs/2409.18556))
- Modern dataset with 12,000 IDE screenshots and pixel-level annotations
- Designed for training object detectors on IDE layouts
- Released July 2024, covers VS Code with 90+ themes

**PsTuts** ([GitHub](https://github.com/KunpengLi1994/PsTuts))
- PyTorch implementation for CVPR 2020 "Screencast Tutorial Video Understanding"
- Includes frame classification and text region detection models
- Can be adapted for custom OCR pipelines

**Pieces** ([Extract Code from Screenshots with OCR](https://pieces.app/features/extract))
- Commercial tool with fine-tuned code OCR
- Specializes in IDE screenshots with context preservation
- Not open-source but demonstrates production-grade approach

**CodeCapture** ([GitHub - KaifHalak/CodeCapture-Chrome-Extension-MV3](https://github.com/KaifHalak/CodeCapture-Chrome-Extension-MV3))
- Chrome extension for manual code region selection from videos
- Uses Tesseract with semi-automatic region cropping
- Simpler alternative to fully automatic detection

**FrameTextExtractor** ([GitHub - zeynelacikgoez/FrameTextExtractor](https://github.com/zeynelacikgoez/FrameTextExtractor))
- Open-source tool combining Tesseract, OpenCV, and multithreading
- Includes motion detection and relevant text filtering
- Practical implementation of frame differencing approach

### Specialized Vision-Language Models

**CLIP (Contrastive Language-Image Pre-training)** ([GitHub - openai/CLIP](https://github.com/openai/CLIP), [OpenAI Blog](https://openai.org/index/clip/))
- Vision-language model that can match images to text descriptions
- Potential application: classify regions as "code", "UI chrome", "documentation" based on visual similarity to code/UI examples
- Can be used for zero-shot region classification without fine-tuning

**CLIP4STR and CLIP-Llama** ([CLIP4STR - arXiv](https://arxiv.org/abs/2305.14014), [CLIP-Llama Paper](https://www.mdpi.com/1424-8220/24/22/7371))
- Variants of CLIP adapted for scene text recognition
- Show promise for recognizing text in challenging conditions (rotated, blurred, irregular)
- Could improve OCR accuracy on IDE code with syntax highlighting

## Part 7: Visual Question Answering (VQA) as Alternative Approach

### Beyond OCR: Semantic Screen Understanding

[VQA: Visual Question Answering](https://visualqa.org/) and the [original ICCV 2015 paper](https://arxiv.org/pdf/1505.00468) present a fundamentally different approach:

Instead of character-level OCR, use a multimodal model to answer questions about screen content:
- Q: "What code is visible in the editor pane?"
- A: [extracted code text with semantic understanding]

This approach is particularly relevant for screencasts because:
1. **Semantic filtering**: VQA inherently understands context (code vs. UI)
2. **Layout agnostic**: Works across different IDE themes and layouts
3. **Intent preservation**: Understands that code is "meaningful" while toolbar text is not

Current VQA models (e.g., LLaVA, GPT-4V) could potentially:
- Classify image regions by semantic role
- Extract text that matches "programming code" semantic patterns
- Ignore regions tagged as "user interface chrome"

This is experimental but represents a promising direction beyond traditional OCR.

## Part 8: Comparative Effectiveness of Approaches

### Noise Reduction Potential

Based on research synthesis, here's expected noise reduction from each approach:

| Approach | Estimated Noise Reduction | Complexity | Accuracy Impact |
|----------|--------------------------|-----------|-----------------|
| Motion detection alone | 60-70% | Low | Neutral (skips frames) |
| Motion + blur detection | 75-85% | Low | Positive (filters blur) |
| CNN region detection (psc2code) | 80-90% | Medium | Very positive (crops to content) |
| CNN + Levenshtein dedup | 85-95% | Medium | Very positive |
| CNN + LLM filtering | 90-98% | High | Very positive (semantic aware) |
| CLIP-based region classification | 85-92% | Medium-High | Positive (zero-shot capable) |

**Important note**: These are estimates based on research descriptions. Actual results depend on:
- IDE theme (light vs. dark reduces detection difficulty)
- Code density (sparse vs. dense code in frame)
- Quality of temporal filtering (tutorial pacing matters)
- Language model training data (if using LLM, must match code language)

## Part 9: Best Practices and Integration Strategy

### Recommended Pipeline for yt-llm-service

Based on research synthesis, a practical multi-stage approach:

**Stage 1: Temporal Filtering** (Quick, high-impact)
- Extract frames at 1 FPS instead of full video
- Apply motion detection; skip frames with <5% change
- Apply FFT blur detection; skip low-quality frames
- Result: Process ~10-15% of original frames

**Stage 2: Region Detection** (CNN or CLIP-based)
- Use CodeSCAN dataset to fine-tune object detector for IDE layouts
- OR use pre-trained CLIP to classify regions ("code editor", "sidebar", "toolbar")
- Crop to content area before OCR
- Result: 80-90% noise reduction at frame level

**Stage 3: OCR with PaddleOCR**
- Use GPU-accelerated PaddleOCR (already available in your system)
- Extract text with confidence scores and bounding boxes
- Filter out bounding boxes that are very small (likely labels) or from UI areas

**Stage 4: Post-Processing**
- Apply Levenshtein distance filtering to remove consecutive duplicates
- Optional: LLM filtering for semantic validation ("is this code/documentation?" vs. "is this UI label?")

### Open Research Questions

- How well do pre-trained CLIP models generalize to IDE screenshots without fine-tuning?
- Can CodeSCAN-trained detectors transfer to other IDEs (Sublime, IntelliJ, Vim) or only VS Code?
- What LLM prompt engineering produces best code/UI distinction on noisy OCR output?
- How important is temporal context (cross-frame consistency) vs. spatial context (region detection)?

## Conclusion

The problem of OCR noise on programming screencasts is well-documented in academic literature, with multiple published solutions. The most effective approaches combine:

1. **Temporal filtering** (motion detection, blur detection) to reduce frame volume
2. **Spatial region detection** (CNN on CodeSCAN, or CLIP-based classification) to isolate content areas
3. **Code-aware post-processing** (LLM filtering, Levenshtein deduplication) to remove remaining UI noise

The psc2code framework and CodeSCAN dataset provide concrete research and tools directly applicable to your problem. For immediate implementation with your existing PaddleOCR infrastructure, prioritize motion detection + region detection + LLM-based semantic filtering as a three-stage pipeline.

---

## Sources

### Academic Papers and Datasets

- [psc2code: Denoising Code Extraction from Programming Screencasts (ICSE 2021 Journal-First)](https://2021.icse-conferences.org/details/icse-2021-Journal-First-Papers/54/psc2code-Denoising-Code-Extraction-from-Programming-Screencasts)
- [psc2code PDF](https://xin-xia.github.io/publication/tosem201.pdf)
- [CodeSCAN: ScreenCast ANalysis for Video Programming Tutorials](https://a-nau.github.io/codescan/)
- [CodeSCAN arXiv Paper](https://arxiv.org/abs/2409.18556)
- [A Study on the Accuracy of OCR Engines for Source Code](https://par.nsf.gov/servlets/purl/10220037)
- [Turning Screenshots into Data: A Four-Level Taxonomy for Screenshot Understanding](https://medium.com/data-science-collective/turning-screensots-int-data-html-126bdcaa4821)
- [Reference-Based Post-OCR Processing with LLM for Precise Diacritic Text](https://arxiv.org/html/2410.13305v1)
- [Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction](https://arxiv.org/html/2510.10138v1)
- [OCR Error Post-Correction with LLMs in Historical Documents](https://arxiv.org/html/2502.01205v1)
- [VQA: Visual Question Answering (ICCV 2015)](https://arxiv.org/pdf/1505.00468)
- [CLIP4STR: A Simple Baseline for Scene Text Recognition](https://arxiv.org/abs/2305.14014)
- [CLIP-Llama: A New Approach for Scene Text Recognition](https://www.mdpi.com/1424-8220/24/22/7371)

### Tools and Projects

- [GitHub - dynobo/normcap: OCR powered screen-capture tool](https://github.com/dynobo/normcap)
- [GitHub - KunpengLi1994/PsTuts: PyTorch code for Screencast Tutorial Video Understanding](https://github.com/KunpengLi1994/PsTuts)
- [GitHub - zeynelacikgoez/FrameTextExtractor: Open-source video frame text extraction tool](https://github.com/zeynelacikgoez/FrameTextExtractor)
- [GitHub - openai/CLIP: Contrastive Language-Image Pre-training](https://github.com/openai/CLIP)
- [GitHub - cofiem/screenshot-ocr: Extract text from screenshots](https://github.com/cofiem/screenshot-ocr)
- [GitHub - KaifHalak/CodeCapture-Chrome-Extension-MV3: Extract Code from Videos Chrome Extension](https://github.com/KaifHalak/CodeCapture-Chrome-Extension-MV3)
- [Pieces: Extract Code from Screenshots with OCR](https://pieces.app/features/extract)
- [CodeCapture Chrome Extension](https://github.com/KaifHalak/CodeCapture-Chrome-Extension-MV3)

### Technical Resources and Guides

- [We Fine-Tuned our OCR to Read Code: Here's What It Took (and What Broke) - DEV Community](https://dev.to/nikl/we-fine-tuned-our-ocr-to-read-code-heres-what-it-took-and-what-broke-4jb8)
- [How to Extract Code from A Screenshot with Accuracy?](https://rikkeisoft.com/blog/extract-code-from-screenshot/)
- [Image to Code OCR Just Released! - Mathpix](https://mathpix.com/blog/image-to-code-ocr)
- [How to Extract Text and OCR from Video Frames with Video Intelligence API](https://oneuptime.com/blog/post/2026-02-17-how-to-extract-text-and-ocr-from-video-frames-with-video-intelligence-api-text-detection/view)
- [Comparing the best methods for OCR on videos - Sieve Blog](https://www.sieve.ai/blog/video-ocr-guide)
- [How to Use OCR on Videos - Roboflow Blog](https://blog.roboflow.com/ocr-on-videos/)
- [OCR'ing Video Streams - PyImageSearch](https://pyimagesearch.com/2022/03/07/ocring-video-streams/)
- [Video OCR - OCR Video to Text Online Free](https://screenapp.io/features/video-ocr)
- [Using LLMs for OCR and PDF Parsing](https://www.cradl.ai/posts/llm-ocr)
- [Large Language Models LLMs for OCR Post-Correction - MarkTechPost](https://www.marktechpost.com/2024/08/13/large-language-models-llms-for-ocr-post-correction/)

### OCR Tool Comparisons

- [PaddleOCR vs Tesseract: Which is the best open source OCR?](https://www.koncile.ai/en/ressources/paddleocr-analyse-avantages-alternatives-open-source)
- [Paddle OCR vs Tesseract: (In-Depth OCR Comparison) - IronOCR](https://ironsoftware.com/csharp/ocr/blog/compare-to-other-components/paddle-ocr-vs-tesseract/)
- [OCR comparison: Tesseract versus EasyOCR vs PaddleOCR vs MMOCR - Medium](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66)
- [Comparing PyTesseract, PaddleOCR, and Surya OCR: Performance on Invoices](https://researchify.io/blog/comparing-pytesseract-paddleocr-and-surya-ocr-performance-on-invoices)

### Vision-Language Models

- [CLIP: Connecting text and images - OpenAI](https://openai.org/index/clip/)
- [Exploring CLIP: A Vision-Language Model for Image Understanding - Medium](https://medium.com/@staytechrich/exploring-clip-a-vision-language-model-vlm-for-image-understanding-9f1e506fe2fd)
- [Mastering the CLIP Vision Model: Key Concepts Revealed](https://www.myscale.com/blog/key-concepts-understand-clip-vision-model/)

### Related Research

- [How Do You Use Deep Learning to Identify UI Components? - Alibaba Cloud](https://www.alibabacloud.com/blog/how-do-you-use-deep-learning-to-identify-ui-components_597859)
- [VQA: Visual Question Answering - visualqa.org](https://visualqa.org/)
- [Extracting code from programming tutorial videos - ML4Code](https://ml4code.github.io/publications/yadid2016extracting/)
