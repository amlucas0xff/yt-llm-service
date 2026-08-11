"""The notes pipeline must find transcript text in every output format.

`--format structured` is the CLI default and returns `blocks`, not `text`.
A notes path that only reads `llm_result["text"]` silently generates nothing
for exactly the format most runs use.
"""
from transcription_service import TranscriptionService


def make_service():
    """Bare instance — extract_transcript_text() touches no instance state."""
    return TranscriptionService.__new__(TranscriptionService)


def test_extracts_plain_text_format():
    svc = make_service()
    assert svc.extract_transcript_text({"text": "hello world"}) == "hello world"


def test_extracts_structured_blocks_format():
    """`structured` output carries blocks — the notes path must not come back empty."""
    svc = make_service()
    result = svc.extract_transcript_text({
        "blocks": [
            {"speaker": "SPEAKER_00", "text": "first thing"},
            {"speaker": "SPEAKER_01", "text": "second thing"},
        ]
    })
    assert "first thing" in result
    assert "second thing" in result


def test_extracts_speaker_map_format():
    svc = make_service()
    result = svc.extract_transcript_text({
        "speakers": {"SPEAKER_00": "alpha", "SPEAKER_01": "beta"}
    })
    assert "alpha" in result
    assert "beta" in result


def test_returns_empty_string_for_unrecognised_shape():
    svc = make_service()
    assert svc.extract_transcript_text({"metadata": {}}) == ""


def test_falls_through_when_text_is_present_but_empty():
    """An empty `text` must not shadow a populated `blocks`/`speakers`."""
    svc = make_service()
    assert "real content" in svc.extract_transcript_text({
        "text": "",
        "blocks": [{"speaker": "SPEAKER_00", "text": "real content"}],
    })
    assert "real content" in svc.extract_transcript_text({
        "text": "",
        "speakers": {"SPEAKER_00": "real content"},
    })
