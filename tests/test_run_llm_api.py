"""Smoke tests for run_llm_api response models."""
from run_llm_api import LLMTranscriptionResponse


def test_llm_transcription_response_has_video_metadata_field():
    """LLMTranscriptionResponse must accept and expose video_metadata."""
    resp = LLMTranscriptionResponse(
        success=True,
        metadata={},
        video_metadata={"title": "Test", "channel": "CH", "tags": [], "categories": [], "description": ""},
    )
    assert resp.video_metadata["title"] == "Test"


def test_llm_transcription_response_video_metadata_defaults_none():
    """video_metadata must default to None when not provided."""
    resp = LLMTranscriptionResponse(success=True, metadata={})
    assert resp.video_metadata is None
