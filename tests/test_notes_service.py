"""Unit tests for NotesService video-context injection."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from audio_downloader import VideoContext
from notes_service import NotesService, TRUNCATION_NOTICE
from config import Config


def make_service():
    cfg = MagicMock(spec=Config)
    cfg.LLAMA_CPP_URL = "http://localhost:8080"
    cfg.NOTES_MAX_TOKENS = 8000
    return NotesService(cfg)


def patched_post(captured_payload):
    """Return an AsyncMock post() that records the JSON body it was called with."""
    async def fake_post(url, json=None, **kwargs):
        captured_payload.update(json)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "# Notes"}}]}
        return resp

    return AsyncMock(side_effect=fake_post)


@pytest.mark.asyncio
async def test_generate_injects_context_block():
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

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = patched_post(captured_payload)
        await svc.generate("raw whisperx transcript", video_context=ctx)

    user_msg = captured_payload["messages"][1]["content"]
    assert "Claude 3.7 Deep Dive" in user_msg
    assert "Anthropic" in user_msg
    assert "claude, anthropic" in user_msg
    assert "raw whisperx transcript" in user_msg


@pytest.mark.asyncio
async def test_generate_without_context_sends_bare_transcript():
    """When no VideoContext is provided, no context block should appear in the prompt."""
    svc = make_service()

    captured_payload = {}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = patched_post(captured_payload)
        await svc.generate("raw whisperx transcript")

    user_msg = captured_payload["messages"][1]["content"]
    assert user_msg == "raw whisperx transcript"
    assert "<video_context>" not in user_msg


def test_correction_api_is_gone():
    """ADR 0001: the dual-ASR correction pass was removed from NotesService."""
    svc = make_service()
    assert not hasattr(svc, "correct_transcript")
    assert not hasattr(svc, "_correct_chunk")


@pytest.mark.asyncio
async def test_generate_reports_no_truncation_for_short_transcript():
    svc = make_service()

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = patched_post({})
        result = await svc.generate("a short transcript")

    assert result.text == "# Notes"
    assert result.truncated is False


@pytest.mark.asyncio
async def test_generate_reports_truncation_for_long_transcript():
    svc = make_service()
    svc.max_tokens = 10  # forces _truncate_transcript to bite

    captured = {}
    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = patched_post(captured)
        result = await svc.generate(" ".join(f"word{i}" for i in range(200)))

    assert result.truncated is True
    assert TRUNCATION_NOTICE.strip() in captured["messages"][1]["content"]


@pytest.mark.asyncio
async def test_generate_on_empty_transcript_returns_empty_result():
    svc = make_service()
    result = await svc.generate("   ")
    assert result.text is None
    assert result.truncated is False
