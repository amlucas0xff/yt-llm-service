"""Tests for ocr_dedup module."""

from ocr_dedup import deduplicate_ocr_results, _jaccard, _to_line_set, _normalize_line  # ty: ignore[unresolved-import]


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
