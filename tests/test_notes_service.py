"""Unit tests for NotesService video-context injection."""
import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
from audio_downloader import VideoContext
from notes_service import NotesService, TRUNCATION_NOTICE
from config import Config


def make_service():
    cfg = MagicMock(spec=Config)
    cfg.LLAMA_CPP_URL = "http://localhost:8080"
    cfg.SIDECAR_GPU_SEPARATE = False
    cfg.LLAMA_CPP_IDLE_SECONDS = 5
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


@pytest.mark.asyncio
async def test_wait_for_gpu_waits_until_llama_is_asleep():
    svc = make_service()
    response = MagicMock()
    response.json.side_effect = [{"is_sleeping": False}, {"is_sleeping": True}]
    with patch("httpx.AsyncClient") as client, patch("notes_service.asyncio.sleep", new_callable=AsyncMock) as pause:
        get = client.return_value.__aenter__.return_value.get
        get.return_value = response
        await svc.wait_for_gpu()

    assert get.await_count == 2
    pause.assert_awaited_once_with(1)


def test_config_reads_explicit_split_marker(monkeypatch):
    monkeypatch.delenv("SIDECAR_GPU_SEPARATE", raising=False)
    assert Config().SIDECAR_GPU_SEPARATE is False
    monkeypatch.setenv("SIDECAR_GPU_SEPARATE", "true")
    assert NotesService(Config()).sidecar_gpu_separate is True


@pytest.mark.asyncio
async def test_wait_for_gpu_skips_sidecar_when_explicitly_split():
    svc = make_service()
    svc.sidecar_gpu_separate = True
    with patch("httpx.AsyncClient") as client:
        await svc.wait_for_gpu()
    client.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_gpu_allows_transcription_if_sidecar_is_down():
    svc = make_service()
    with patch("httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError("offline")
        await svc.wait_for_gpu()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [300, -1, 30])
async def test_wait_for_gpu_rejects_invalid_shared_idle_before_props(idle):
    svc = make_service()
    svc.idle_seconds = idle
    with patch("httpx.AsyncClient") as client:
        with pytest.raises(RuntimeError, match="LLAMA_CPP_IDLE_SECONDS"):
            await svc.wait_for_gpu()
    client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("read timed out"),
    httpx.HTTPStatusError("500", request=httpx.Request("GET", "http://localhost:8080/props"), response=httpx.Response(500)),
    ValueError("invalid JSON"),
])
async def test_wait_for_gpu_rejects_reachable_props_errors(failure):
    svc = make_service()
    response = MagicMock(status_code=500)
    if isinstance(failure, httpx.HTTPStatusError):
        response.raise_for_status.side_effect = failure
    elif isinstance(failure, ValueError):
        response.status_code = 200
        response.json.side_effect = failure
    with patch("httpx.AsyncClient") as client:
        get = client.return_value.__aenter__.return_value.get
        if isinstance(failure, httpx.ReadTimeout):
            get.side_effect = failure
        else:
            get.return_value = response
        with pytest.raises((httpx.HTTPError, ValueError)):
            await svc.wait_for_gpu()


@pytest.mark.asyncio
async def test_wait_for_gpu_rejects_props_without_sleep_state():
    svc = make_service()
    response = MagicMock(status_code=200)
    response.json.return_value = {}
    with patch("httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.get.return_value = response
        with pytest.raises(RuntimeError, match="is_sleeping"):
            await svc.wait_for_gpu()


@pytest.mark.asyncio
async def test_wait_for_gpu_allows_sidecar_down_even_with_valid_shared_idle():
    svc = make_service()
    with patch("httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError("offline")
        await svc.wait_for_gpu()


@pytest.mark.asyncio
async def test_wait_for_gpu_fails_if_sidecar_never_unloads():
    svc = make_service()
    response = MagicMock()
    response.json.return_value = {"is_sleeping": False}
    with patch("httpx.AsyncClient") as client, patch("notes_service.asyncio.sleep", new_callable=AsyncMock):
        get = client.return_value.__aenter__.return_value.get
        get.return_value = response
        with pytest.raises(RuntimeError, match="still using the GPU"):
            await svc.wait_for_gpu()

    assert get.await_count == 30


@pytest.mark.asyncio
async def test_wait_for_gpu_retries_while_sidecar_is_loading():
    svc = make_service()
    loading = MagicMock(status_code=503)
    loading.raise_for_status.side_effect = httpx.HTTPStatusError(
        "loading", request=httpx.Request("GET", "http://localhost:8080/props"), response=httpx.Response(503)
    )
    asleep = MagicMock(status_code=200)
    asleep.json.return_value = {"is_sleeping": True}
    with patch("httpx.AsyncClient") as client, patch("notes_service.asyncio.sleep", new_callable=AsyncMock):
        get = client.return_value.__aenter__.return_value.get
        get.side_effect = [loading, asleep]
        await svc.wait_for_gpu()

    assert get.await_count == 2
