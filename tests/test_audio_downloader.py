"""Unit tests for AudioDownloader.get_video_context()"""
from unittest.mock import patch, MagicMock
from audio_downloader import AudioDownloader, VideoContext

FAKE_INFO = {
    "id": "abc123",
    "title": "Claude 3.7 Sonnet Deep Dive",
    "description": "Anthropic releases Claude 3.7 Sonnet with extended thinking.",
    "channel": "Anthropic",
    "uploader": "Anthropic",
    "tags": ["claude", "anthropic", "llm"],
    "categories": ["Science & Technology"],
}


def make_downloader(tmp_path):
    with patch("audio_downloader.shutil.which", return_value="/usr/bin/yt-dlp"):
        return AudioDownloader(temp_dir=str(tmp_path))


def test_get_video_context_returns_dataclass(tmp_path):
    dl = make_downloader(tmp_path)
    with patch("yt_dlp.YoutubeDL") as MockYDL:
        instance = MockYDL.return_value.__enter__.return_value
        instance.extract_info.return_value = FAKE_INFO

        ctx = dl.get_video_context("https://www.youtube.com/watch?v=abc123")

    assert isinstance(ctx, VideoContext)
    assert ctx.video_id == "abc123"
    assert ctx.title == "Claude 3.7 Sonnet Deep Dive"
    assert ctx.channel == "Anthropic"
    assert ctx.tags == ["claude", "anthropic", "llm"]
    assert ctx.categories == ["Science & Technology"]
    assert "Anthropic releases" in ctx.description


def test_get_video_context_never_raises_on_error(tmp_path):
    dl = make_downloader(tmp_path)
    with patch("yt_dlp.YoutubeDL", side_effect=Exception("network failure")):
        ctx = dl.get_video_context("https://www.youtube.com/watch?v=abc123")

    assert isinstance(ctx, VideoContext)
    assert ctx.title == ""
    assert ctx.video_id == "abc123"


def test_video_context_no_longer_carries_captions():
    """ADR 0001: captions existed only to feed the dual-ASR correction pass."""
    assert not hasattr(VideoContext(), "captions")


def test_download_audio_result_includes_video_context(tmp_path):
    """download_audio() result dict must include 'video_context' key with VideoContext."""
    dl = make_downloader(tmp_path)

    # Create a fake mp3 so the file-exists check passes
    fake_mp3 = tmp_path / "abc123.mp3"
    fake_mp3.write_bytes(b"fake")

    with patch.object(dl, "get_video_context") as mock_ctx, \
         patch.object(dl, "_cleanup_old_files"), \
         patch("subprocess.run") as mock_run:

        mock_ctx.return_value = VideoContext(
            video_id="abc123",
            title="Test Video",
            channel="TestChannel",
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = dl.download_audio("https://www.youtube.com/watch?v=abc123")

    assert "video_context" in result
    assert isinstance(result["video_context"], VideoContext)
    assert result["title"] == "Test Video"
