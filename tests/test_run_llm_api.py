"""Smoke tests for run_llm_api response models."""
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import httpx
from notes_service import NotesService

import pytest

from run_llm_api import LLMTranscriptionResponse, transcribe_with_gpu_release


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


@pytest.mark.asyncio
async def test_transcription_waits_for_sidecar_before_using_gpu():
    events = []
    with patch("run_llm_api.notes_service") as notes, patch("run_llm_api.transcription_service") as transcription:
        notes.wait_for_gpu = AsyncMock(side_effect=lambda: events.append("wait"))
        transcription.device = "cuda"
        transcription.transcribe_audio.side_effect = lambda **kwargs: events.append("transcribe") or {"segments": []}

        result = await transcribe_with_gpu_release(audio_path="/app/tmp/video.mp3")

    assert result == {"segments": []}
    assert events == ["wait", "transcribe"]


@pytest.mark.asyncio
async def test_cpu_transcription_does_not_wait_for_sidecar():
    with patch("run_llm_api.notes_service") as notes, patch("run_llm_api.transcription_service") as transcription:
        transcription.device = "cpu"
        transcription.transcribe_audio.return_value = {"segments": []}

        await transcribe_with_gpu_release(audio_path="/app/tmp/video.mp3")

    notes.wait_for_gpu.assert_not_called()
    transcription.transcribe_audio.assert_called_once_with(audio_path="/app/tmp/video.mp3")

def gpu_notes(idle, split=False):
    return NotesService(SimpleNamespace(
        LLAMA_CPP_URL="http://llama-cpp:8080", SIDECAR_GPU_SEPARATE=split,
        LLAMA_CPP_IDLE_SECONDS=idle, NOTES_MAX_TOKENS=8000,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [300, -1])
async def test_request_rejects_invalid_shared_idle_before_cuda_even_if_props_sleeps(idle):
    with patch("run_llm_api.notes_service", gpu_notes(idle)), \
         patch("run_llm_api.transcription_service") as transcription, \
         patch("httpx.AsyncClient") as client:
        transcription.device = "cuda"
        client.return_value.__aenter__.return_value.get.return_value.json.return_value = {"is_sleeping": True}
        with pytest.raises(RuntimeError, match="LLAMA_CPP_IDLE_SECONDS"):
            await transcribe_with_gpu_release(audio_path="/app/tmp/video.mp3")
        client.assert_not_called()
        transcription.transcribe_audio.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("split,idle", [(False, 5), (True, -1)])
async def test_request_allows_optional_sidecar_down_or_explicit_split(split, idle):
    with patch("run_llm_api.notes_service", gpu_notes(idle, split)), \
         patch("run_llm_api.transcription_service") as transcription, \
         patch("httpx.AsyncClient") as client:
        transcription.device = "cuda"
        client.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError("offline")
        transcription.transcribe_audio.return_value = {"segments": []}
        assert await transcribe_with_gpu_release(audio_path="/app/tmp/video.mp3") == {"segments": []}
        transcription.transcribe_audio.assert_called_once()
        if split:
            client.assert_not_called()
