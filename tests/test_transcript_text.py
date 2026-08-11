"""The notes pipeline must find transcript text in every output format.

`--format structured` is the CLI default and returns `blocks`, not `text`.
A notes path that only reads `llm_result["text"]` silently generates nothing
for exactly the format most runs use.
"""
from unittest.mock import MagicMock

from transcription_service import TranscriptionService
from config import Config


def make_service():
    cfg = MagicMock(spec=Config)
    cfg.DEVICE = "cpu"
    cfg.COMPUTE_TYPE = "float32"
    cfg.WHISPER_MODEL = "tiny"
    cfg.BATCH_SIZE = 1
    cfg.HF_TOKEN = None
    cfg.OUTPUT_DIR = "/tmp"
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
