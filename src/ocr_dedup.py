"""Post-OCR deduplication -- merge consecutive frames with overlapping text.

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
