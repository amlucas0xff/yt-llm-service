"""Tests for cli.py input detection and output rendering logic."""
import pytest
from unittest.mock import MagicMock


def test_detect_youtube_url():
    from cli import detect_input_type
    assert detect_input_type("https://www.youtube.com/watch?v=abc123") == "youtube"
    assert detect_input_type("https://youtu.be/abc123") == "youtube"


def test_detect_local_file():
    from cli import detect_input_type
    assert detect_input_type("/tmp/video.mp4") == "file"
    assert detect_input_type("./recording.m4a") == "file"


def test_blocks_to_markdown_single_speaker():
    from cli import blocks_to_markdown
    blocks = [{"speaker": "SPEAKER_00", "text": "Hello world."}]
    result = blocks_to_markdown(blocks)
    assert "**SPEAKER_00:**" in result
    assert "Hello world." in result


def test_blocks_to_markdown_multiple_speakers():
    from cli import blocks_to_markdown
    blocks = [
        {"speaker": "SPEAKER_00", "text": "First line."},
        {"speaker": "SPEAKER_01", "text": "Second line."},
        {"speaker": "SPEAKER_00", "text": "Third line."},
    ]
    result = blocks_to_markdown(blocks)
    assert result.count("**SPEAKER_00:**") == 2
    assert result.count("**SPEAKER_01:**") == 1


def test_blocks_to_markdown_no_speaker():
    from cli import blocks_to_markdown
    blocks = [{"text": "No speaker info here."}]
    result = blocks_to_markdown(blocks)
    assert "No speaker info here." in result


def test_build_youtube_payload():
    from cli import build_youtube_payload
    payload = build_youtube_payload(
        url="https://youtu.be/abc",
        fmt="structured",
        notes=True,
        no_filler=False,
        min_speakers=None,
        max_speakers=None,
    )
    assert payload["youtube_url"] == "https://youtu.be/abc"
    assert payload["output_format"] == "structured"
    assert payload["generate_notes"] is True
    assert payload["remove_filler_words"] is False
    assert "min_speakers" not in payload  # None values should be excluded


def test_build_file_fields():
    from cli import build_file_fields
    fields = build_file_fields(
        fmt="simple",
        notes=False,
        no_filler=True,
        min_speakers=2,
        max_speakers=4,
    )
    assert fields["output_format"] == "simple"
    assert fields["generate_notes"] == "false"
    assert fields["remove_filler_words"] == "true"
    assert fields["min_speakers"] == "2"
    assert fields["max_speakers"] == "4"


def make_client(status_code, json_body=None, text=""):
    """httpx.Client double whose post() returns a canned response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body if json_body is not None else {}
    client = MagicMock()
    client.post.return_value = resp
    return client, resp


def test_post_youtube_calls_sync_endpoint_and_returns_payload():
    from cli import post_youtube

    client, _ = make_client(200, {"success": True, "text": "Hello world"})
    payload = {"youtube_url": "https://youtu.be/abc"}

    result = post_youtube(client, "http://localhost:8002", payload)

    client.post.assert_called_once_with(
        "http://localhost:8002/transcribe-youtube-llm", json=payload
    )
    assert result["text"] == "Hello world"


def test_post_youtube_exits_on_service_error():
    from cli import post_youtube
    import typer

    client, _ = make_client(500, {"detail": "yt-dlp failed"}, text="yt-dlp failed")

    with pytest.raises(typer.Exit) as exc:
        post_youtube(client, "http://localhost:8002", {"youtube_url": "https://youtu.be/abc"})

    assert exc.value.exit_code == 1


def test_post_youtube_exits_when_error_body_is_not_json():
    from cli import post_youtube
    import typer

    client, resp = make_client(502, text="<html>bad gateway</html>")
    resp.json.side_effect = ValueError("not json")

    with pytest.raises(typer.Exit) as exc:
        post_youtube(client, "http://localhost:8002", {"youtube_url": "https://youtu.be/abc"})

    assert exc.value.exit_code == 1
