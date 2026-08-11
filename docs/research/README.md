# OCR Screencast Research Documentation

This directory contains comprehensive research on solving the OCR noise problem in programming screencast video text extraction.

## Quick Start

Start here based on your role:

### I'm an Executive/Decision Maker
Read: **COMPLETION_SUMMARY.md** (5-10 minutes)
- Problem statement and scope
- Solution overview and expected outcomes
- Timeline and effort
- Top 3 critical insights

### I'm a Technical Architect/Planner
Read in order:
1. **RESEARCH_INDEX.md** (10 minutes) — Overview and decision guide
2. **ocr-screencast-actionable-recommendations.md** (30 minutes) — Implementation approach
3. **ocr-screencast-noise-problem.md** — Reference as needed for deep understanding

### I'm an Engineer Starting Implementation
Read in order:
1. **RESEARCH_INDEX.md** — Understand problem and approaches
2. **ocr-screencast-actionable-recommendations.md** — Follow step-by-step implementation guidance with code examples
3. **ocr-screencast-noise-problem.md** — Reference specific techniques or algorithms as needed

### I Need Everything (Deep Dive)
Read in order:
1. COMPLETION_SUMMARY.md
2. RESEARCH_INDEX.md
3. ocr-screencast-actionable-recommendations.md
4. ocr-screencast-noise-problem.md

## Document Overview

### COMPLETION_SUMMARY.md
**Purpose**: Executive summary and research completion report
**Length**: 308 lines
**Time**: 5-10 minutes
**Content**:
- Overview and research quality summary
- Key findings and discoveries
- Integration considerations
- Timeline and testing strategy
- Top 3 critical insights

### RESEARCH_INDEX.md
**Purpose**: Quick-reference guide tying all research together
**Length**: 217 lines
**Time**: 10 minutes
**Content**:
- Problem statement
- Document structure
- Tool and dataset reference table
- Expected outcomes by implementation phase
- Decision guide (which approach to start with)
- Questions and next actions

### ocr-screencast-actionable-recommendations.md
**Purpose**: Step-by-step implementation guidance with code examples
**Length**: 605 lines
**Time**: 30 minutes
**Content**:
- Recommendation 1: Temporal Filtering (2-3 days, Python code examples)
- Recommendation 2: Region-of-Interest Detection (5-7 days, 3 implementation options with code)
- Recommendation 3: LLM-Based Post-Processing (3-5 days, integration with existing services)
- Implementation roadmap with phases
- Configuration defaults and validation metrics
- Risk mitigation strategies

**Best for**: Implementation planning, coding reference, testing setup

### ocr-screencast-noise-problem.md
**Purpose**: Deep-dive technical research on all approaches
**Length**: 388 lines, 40+ citations
**Time**: 1 hour for full deep-dive, sections can be read independently
**Content**:
- Part 1: Why screenshot OCR captures all UI layers
- Part 2: Region-of-Interest detection techniques
- Part 3: Temporal filtering approaches
- Part 4: Code-specific OCR and accuracy comparisons
- Part 5: LLM-based post-processing
- Part 6: Existing tools and projects (psc2code, CodeSCAN, CLIP, etc.)
- Part 7: Visual Question Answering as alternative approach
- Part 8: Comparative effectiveness table
- Part 9: Integration strategy and best practices
- Complete citations and source links

**Best for**: Deep understanding, technical validation, architectural decisions

## The Problem in 30 Seconds

Your OCR system extracts text from programming tutorial video frames and gets:
- **1093 lines of output** for a few frames
- **Signal-to-noise ratio**: 1:150 (only 2 meaningful lines per 300 lines total)
- **Root cause**: OCR captures ALL text equally—code, toolbar, tabs, menu, status bar, sidebar, keyboard overlays

## The Solution in 30 Seconds

Implement a 3-stage pipeline:
1. **Temporal filtering** (motion + blur detection): Skip 85-90% of frames with zero accuracy loss
2. **Spatial filtering** (CLIP-based region detection): Crop to code area, run OCR only on content region
3. **LLM post-processing** (semantic filtering): Remove remaining UI labels, correct OCR errors

Expected outcome: **90-98% noise reduction**, output ~15-20 meaningful lines instead of 1093

Timeline: **10-15 days** to full implementation (5-7 days if parallelized)

## Key Discoveries

### Academic Research Directly Applicable
- **psc2code** (ICSE 2021) — "Denoising Code Extraction from Programming Screencasts" — exact solution to your problem
- **CodeSCAN dataset** (2024) — 12,000 annotated IDE screenshots for training detectors

### Recommended Approach
- **Start with temporal filtering** (2-3 days, 10x speed improvement)
- **Add CLIP region detection** (5-7 days, 80-90% noise reduction per frame)
- **Add LLM filtering** (3-5 days, final 90-98% noise reduction)

### Why This Works
- Temporal: Static tutorial screens don't need OCR (motion detection skips them)
- Spatial: CLIP can classify "code editor" vs. "UI chrome" without training
- LLM: Your existing llama-cpp can semantically filter OCR output

## Research Quality

- **8-10 hours** of research across 35+ web searches
- **40+ sources** cited: academic papers, open-source projects, technical guides
- **3 implementation phases** with code examples
- **Testing strategies** for each phase
- **Risk mitigation** for known issues
- **Integration points** mapped to your existing code

## Next Steps

1. **Decide**: Which role above matches you? Start with the appropriate document.
2. **Review**: Read recommended documents with your team.
3. **Plan**: Use RESEARCH_INDEX.md decision guide to pick starting approach.
4. **Implement**: Follow ocr-screencast-actionable-recommendations.md step-by-step.
5. **Validate**: Test with your screencast samples, measure noise reduction metrics.
6. **Document**: Record architectural decisions in project ADRs.

## Questions?

Each document includes:
- **COMPLETION_SUMMARY.md**: Open questions for your team
- **RESEARCH_INDEX.md**: Decision guide and next actions
- **ocr-screencast-actionable-recommendations.md**: Risk mitigation and configuration defaults
- **ocr-screencast-noise-problem.md**: Complete citations and source references

All sources are linked directly to original materials for fact-checking or deeper research.

---

**Research completed**: 2026-03-05
**Scope**: OCR text extraction from programming screencast video frames
**Problem**: 1093 lines of noise for a few frames
**Target**: 15-20 lines of meaningful content (90-98% noise reduction)
**Timeline**: 10-15 days to full solution
