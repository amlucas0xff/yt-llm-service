"""Unit tests for NotesService GEC context injection."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from audio_downloader import VideoContext
from notes_service import NotesService
from config import Config


def make_service():
    cfg = MagicMock(spec=Config)
    cfg.LLAMA_CPP_URL = "http://localhost:8080"
    cfg.NOTES_MAX_TOKENS = 8000
    return NotesService(cfg)


@pytest.mark.asyncio
async def test_correct_transcript_injects_context_block():
    """When VideoContext is provided, the user message must contain the context block."""
    svc = make_service()
    ctx = VideoContext(
        video_id="abc123",
        title="Claude 3.7 Deep Dive",
        channel="Anthropic",
        tags=["claude", "anthropic"],
        categories=["Science & Technology"],
        description="Anthropic releases Claude 3.7 Sonnet.",
    )

    captured_payload = {}

    async def fake_post(url, json=None, **kwargs):
        captured_payload.update(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "corrected text"}}]
        }
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await svc.correct_transcript("raw whisperx", "yt captions", video_context=ctx)

    user_msg = captured_payload["messages"][1]["content"]
    assert "Claude 3.7 Deep Dive" in user_msg
    assert "Anthropic" in user_msg
    assert "claude, anthropic" in user_msg


@pytest.mark.asyncio
async def test_correct_transcript_without_context_omits_block():
    """When no VideoContext is provided, no context block should appear in the prompt."""
    svc = make_service()

    captured_payload = {}

    async def fake_post(url, json=None, **kwargs):
        captured_payload.update(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "corrected text"}}]
        }
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await svc.correct_transcript("raw whisperx", "yt captions")

    user_msg = captured_payload["messages"][1]["content"]
    assert "## Video Context" not in user_msg
